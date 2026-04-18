"""Stage 117 — SecureVault: encrypted local secrets store.

Stores API keys, passwords, and tokens locally using Fernet symmetric
encryption from the ``cryptography`` library.  Falls back to base64
obfuscation with a warning when ``cryptography`` is not installed.

The encryption key is derived from a user-supplied passphrase (and
optionally the machine ID).

Usage
-----
::

    from physml.secure_vault import SecureVault

    vault = SecureVault(vault_path="~/.mycelium/vault.enc")
    vault.unlock("my-passphrase")
    vault.set("openai_key", "sk-...")
    vault.get("openai_key")    # → "sk-..."
    vault.list_keys()          # → ["openai_key"]
    vault.lock()
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

from physml._log import get_logger

_logger = get_logger(__name__)

# Try cryptography (Fernet)
try:
    from cryptography.fernet import Fernet  # type: ignore
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC  # type: ignore
    from cryptography.hazmat.primitives import hashes  # type: ignore

    _CRYPTO = True
except ImportError:
    _CRYPTO = False
    _logger.warning(
        "SecureVault: 'cryptography' not installed; using base64 obfuscation "
        "(NOT secure). Install with: pip install cryptography"
    )

_SALT_SIZE = 16
_ITERATIONS = 100_000


def _derive_key(passphrase: str, salt: bytes) -> bytes:
    """Derive a 32-byte key from *passphrase* + *salt*."""
    if _CRYPTO:
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=_ITERATIONS,
        )
        return base64.urlsafe_b64encode(kdf.derive(passphrase.encode("utf-8")))
    else:
        # Weak fallback: SHA-256 of passphrase+salt
        h = hashlib.sha256(passphrase.encode("utf-8") + salt).digest()
        return base64.urlsafe_b64encode(h)


def _get_machine_id() -> str:
    """Return a stable machine identifier string."""
    try:
        with open("/etc/machine-id", "r") as f:
            return f.read().strip()
    except Exception:
        pass
    try:
        with open("/var/lib/dbus/machine-id", "r") as f:
            return f.read().strip()
    except Exception:
        pass
    return "mycelium-default-machine"


class SecureVault:
    """Encrypted local key-value secrets store.

    Parameters
    ----------
    vault_path : str, default "~/.mycelium/vault.enc"
        Path to the encrypted vault file.
    use_machine_id : bool, default True
        When ``True``, mixes the machine ID into the key derivation for
        extra binding to the local device.
    """

    def __init__(
        self,
        vault_path: str = "~/.mycelium/vault.enc",
        use_machine_id: bool = True,
    ) -> None:
        self.vault_path = Path(vault_path).expanduser()
        self.use_machine_id = use_machine_id
        self._unlocked = False
        self._data: Dict[str, str] = {}
        self._fernet: Any = None  # Fernet instance or None
        self._salt: Optional[bytes] = None

    # ------------------------------------------------------------------
    # Lock / unlock
    # ------------------------------------------------------------------

    def unlock(self, passphrase: str) -> None:
        """Unlock the vault with *passphrase*.

        Loads and decrypts the vault file if it exists; creates a new
        empty vault otherwise.

        Parameters
        ----------
        passphrase : str
        """
        if self.use_machine_id:
            passphrase = passphrase + _get_machine_id()

        if self.vault_path.exists():
            raw = self.vault_path.read_bytes()
            self._salt = raw[:_SALT_SIZE]
            key = _derive_key(passphrase, self._salt)
            try:
                self._data = self._decrypt(raw[_SALT_SIZE:], key)
                self._fernet_key = key
                self._unlocked = True
                _logger.info("SecureVault: unlocked %s", self.vault_path)
            except Exception as exc:
                self._unlocked = False
                raise ValueError(f"SecureVault: failed to unlock (wrong passphrase?): {exc}") from exc
        else:
            # New vault
            self._salt = os.urandom(_SALT_SIZE)
            self._fernet_key = _derive_key(passphrase, self._salt)
            self._data = {}
            self._unlocked = True
            _logger.info("SecureVault: created new vault at %s", self.vault_path)

    def lock(self) -> None:
        """Save and lock the vault, clearing in-memory data."""
        if self._unlocked:
            self._flush()
        self._data = {}
        self._fernet = None
        self._fernet_key = None
        self._unlocked = False
        _logger.info("SecureVault: locked")

    def _require_unlocked(self) -> None:
        if not self._unlocked:
            raise RuntimeError("SecureVault is locked. Call unlock() first.")

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def set(self, key: str, value: str) -> None:
        """Store a secret.

        Parameters
        ----------
        key : str
        value : str
        """
        self._require_unlocked()
        self._data[key] = value
        self._flush()

    def get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Retrieve a secret.

        Parameters
        ----------
        key : str
        default : str, optional

        Returns
        -------
        str or None
        """
        self._require_unlocked()
        return self._data.get(key, default)

    def delete(self, key: str) -> bool:
        """Remove a secret.

        Parameters
        ----------
        key : str

        Returns
        -------
        bool
            ``True`` if the key existed.
        """
        self._require_unlocked()
        existed = key in self._data
        self._data.pop(key, None)
        if existed:
            self._flush()
        return existed

    # Convenience aliases
    def store(self, key: str, value: str) -> None:
        """Alias for :meth:`set`."""
        self.set(key, value)

    def retrieve(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Alias for :meth:`get`."""
        return self.get(key, default)

    def save(self) -> None:
        """Flush current in-memory vault to disk (alias for internal _flush)."""
        self._require_unlocked()
        self._flush()

    def load(self) -> None:
        """Reload vault from disk (re-decrypt with current key)."""
        self._require_unlocked()
        if self.vault_path.exists():
            raw = self.vault_path.read_bytes()
            self._salt = raw[:_SALT_SIZE]
            self._data = self._decrypt(raw[_SALT_SIZE:], self._fernet_key)

    def list_keys(self) -> List[str]:
        """Return all stored secret names.

        Returns
        -------
        list of str
        """
        self._require_unlocked()
        return list(self._data.keys())

    # ------------------------------------------------------------------
    # Encryption helpers
    # ------------------------------------------------------------------

    def _flush(self) -> None:
        """Encrypt and write the vault to disk."""
        self.vault_path.parent.mkdir(parents=True, exist_ok=True)
        ciphertext = self._encrypt(self._data, self._fernet_key)
        self.vault_path.write_bytes(self._salt + ciphertext)

    def _encrypt(self, data: Dict[str, str], key: bytes) -> bytes:
        plain = json.dumps(data).encode("utf-8")
        if _CRYPTO:
            f = Fernet(key)
            return f.encrypt(plain)
        else:
            warnings.warn(
                "SecureVault: using base64 obfuscation (NOT secure). "
                "Install 'cryptography' for real encryption.",
                stacklevel=3,
            )
            return base64.urlsafe_b64encode(plain)

    def _decrypt(self, ciphertext: bytes, key: bytes) -> Dict[str, str]:
        if _CRYPTO:
            f = Fernet(key)
            plain = f.decrypt(ciphertext)
        else:
            plain = base64.urlsafe_b64decode(ciphertext)
        return json.loads(plain.decode("utf-8"))

    def __repr__(self) -> str:
        status = "unlocked" if self._unlocked else "locked"
        return f"SecureVault(path={self.vault_path}, status={status})"
