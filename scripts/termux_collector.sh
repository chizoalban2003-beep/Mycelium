#!/data/data/com.termux/files/usr/bin/bash
# Myco — Android Signal Collector via Termux
#
# Install Termux from F-Droid, then:
#   pkg install termux-api curl jq
#   chmod +x termux_collector.sh
#   ./termux_collector.sh
#
# This script collects phone signals and posts them to your Myco server.
# Set MYCO_URL to your server's address (laptop IP or cloud URL).

MYCO_URL="${MYCO_URL:-http://192.168.1.100:8000}"
MYCO_EMAIL="${MYCO_EMAIL:-}"
MYCO_PASSWORD="${MYCO_PASSWORD:-}"
INTERVAL="${INTERVAL:-30}"

echo ""
echo "  🌱 Myco Android Signal Collector"
echo "  Server: $MYCO_URL"
echo "  Interval: ${INTERVAL}s"
echo ""

# Login
if [ -z "$MYCO_TOKEN" ]; then
  if [ -z "$MYCO_EMAIL" ] || [ -z "$MYCO_PASSWORD" ]; then
    echo "Set MYCO_EMAIL and MYCO_PASSWORD, or MYCO_TOKEN"
    exit 1
  fi
  MYCO_TOKEN=$(curl -sS -X POST "${MYCO_URL}/api/auth/login" \
    -H 'Content-Type: application/x-www-form-urlencoded' \
    --data-urlencode "username=${MYCO_EMAIL}" \
    --data-urlencode "password=${MYCO_PASSWORD}" \
    | jq -r '.access_token // empty')

  if [ -z "$MYCO_TOKEN" ]; then
    echo "Login failed"
    exit 1
  fi
  echo "  Logged in"
fi

AUTH="Authorization: Bearer $MYCO_TOKEN"

post_signal() {
  local sig_type="$1"
  local payload="$2"
  curl -sS -X POST "${MYCO_URL}/api/nexus/stimulus" \
    -H "$AUTH" \
    -H "Content-Type: application/json" \
    -d "{
      \"device_id\": \"android\",
      \"source\": \"termux\",
      \"modality\": \"telemetry\",
      \"signal_type\": \"${sig_type}\",
      \"stimulus\": ${payload}
    }" > /dev/null 2>&1
}

echo "  Collecting signals..."
echo ""

while true; do
  # Battery
  BATT=$(termux-battery-status 2>/dev/null | jq -c '{
    battery_percent: .percentage,
    battery_plugged: (.plugged != "UNPLUGGED"),
    battery_status: .status,
    battery_temperature: .temperature
  }' 2>/dev/null || echo '{}')

  if [ "$BATT" != "{}" ]; then
    post_signal "battery_status" "$BATT"
    echo "  $(date +%H:%M:%S) battery: $(echo $BATT | jq -r '.battery_percent')%"
  fi

  # WiFi
  WIFI=$(termux-wifi-connectioninfo 2>/dev/null | jq -c '{
    wifi_ssid: .ssid,
    wifi_rssi: .rssi,
    wifi_link_speed: .link_speed_mbps,
    wifi_frequency: .frequency_mhz
  }' 2>/dev/null || echo '{}')

  if [ "$WIFI" != "{}" ]; then
    post_signal "wifi_status" "$WIFI"
  fi

  # Screen brightness (proxy for screen state)
  BRIGHT=$(termux-brightness 2>/dev/null || echo "")

  # Audio state
  AUDIO=$(termux-audio-info 2>/dev/null | jq -c '{
    music_active: .MUSIC_ACTIVE,
    volume_music: .STREAM_MUSIC_VOLUME
  }' 2>/dev/null || echo '{}')

  if [ "$AUDIO" != "{}" ]; then
    post_signal "audio_state" "$AUDIO"
  fi

  # Location (if permitted)
  LOC=$(termux-location -p passive -r last 2>/dev/null | jq -c '{
    latitude: .latitude,
    longitude: .longitude,
    altitude: .altitude,
    accuracy: .accuracy
  }' 2>/dev/null || echo '{}')

  if [ "$LOC" != "{}" ] && [ "$(echo $LOC | jq '.latitude')" != "null" ]; then
    post_signal "location" "$LOC"
  fi

  # Sensor snapshot (accelerometer — movement detection)
  SENSOR=$(termux-sensor -s "accelerometer" -n 1 2>/dev/null | jq -c '
    if .accelerometer then
      {accel_x: .accelerometer.values[0], accel_y: .accelerometer.values[1], accel_z: .accelerometer.values[2]}
    else {} end
  ' 2>/dev/null || echo '{}')

  if [ "$SENSOR" != "{}" ]; then
    post_signal "motion" "$SENSOR"
  fi

  # Also trigger server-side collection
  curl -sS -X POST "${MYCO_URL}/api/ecosystem/collect" -H "$AUTH" > /dev/null 2>&1

  sleep "$INTERVAL"
done
