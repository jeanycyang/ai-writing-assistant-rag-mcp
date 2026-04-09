#!/usr/bin/env bash
set -euo pipefail

if [[ -f ".env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source ".env"
  set +a
fi

ACTION="${1:-up}"
TARGET_HOST="${TAILSCALE_FUNNEL_TARGET_HOST:-127.0.0.1}"
TARGET_PORT="${TAILSCALE_FUNNEL_TARGET_PORT:-${RAG_API_PORT:-8001}}"
TARGET_URL="${TAILSCALE_FUNNEL_TARGET_URL:-http://${TARGET_HOST}:${TARGET_PORT}}"
HTTPS_PORT="${TAILSCALE_FUNNEL_HTTPS_PORT:-443}"
MCP_PATH="${TAILSCALE_MCP_PATH:-/mcp}"

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

get_dns_name() {
  tailscale status --json | python3 -c 'import json, sys; print(json.load(sys.stdin).get("Self", {}).get("DNSName", "").rstrip("."))'
}

format_base_url() {
  local dns_name="$1"
  if [[ "$HTTPS_PORT" == "443" ]]; then
    printf 'https://%s' "$dns_name"
    return
  fi
  printf 'https://%s:%s' "$dns_name" "$HTTPS_PORT"
}

print_urls() {
  local dns_name
  dns_name="$(get_dns_name)"
  if [[ -z "$dns_name" ]]; then
    echo "Unable to determine the device DNS name from tailscale status." >&2
    return 1
  fi

  local base_url
  base_url="$(format_base_url "$dns_name")"
  echo "Funnel URL: $base_url"
  echo "MCP endpoint: ${base_url}${MCP_PATH}"
}

require_cmd tailscale
require_cmd python3

case "$ACTION" in
  up)
    tailscale funnel --bg --https="$HTTPS_PORT" "$TARGET_URL"
    print_urls || true
    tailscale funnel status
    ;;
  status)
    print_urls || true
    tailscale funnel status
    ;;
  url)
    print_urls
    ;;
  down|reset)
    tailscale funnel reset
    ;;
  *)
    echo "Usage: $0 [up|status|url|down]" >&2
    exit 1
    ;;
esac
