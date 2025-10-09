#!/usr/bin/env bash
# Permanently grant current user access to Orbbec (VID=2bc5) cameras
# Works on Ubuntu/Debian family; requires sudo/root to write udev rules.

set -euo pipefail

# Determine target username (supports sudo invocation)
TARGET_USER="${SUDO_USER:-$USER}"

if [[ $EUID -ne 0 ]]; then
  echo "Please run with sudo:  sudo bash $0"
  exit 1
fi

echo ">>> Target user: ${TARGET_USER}"

# 1) Create dedicated group
GROUP_NAME="orbbec"
if getent group "${GROUP_NAME}" >/dev/null; then
  echo ">>> Group '${GROUP_NAME}' already exists."
else
  echo ">>> Creating group '${GROUP_NAME}' ..."
  groupadd "${GROUP_NAME}"
fi

# 2) Add user to group
if id -nG "${TARGET_USER}" | tr ' ' '\n' | grep -qx "${GROUP_NAME}"; then
  echo ">>> User '${TARGET_USER}' is already in group '${GROUP_NAME}'."
else
  echo ">>> Adding '${TARGET_USER}' to group '${GROUP_NAME}' ..."
  usermod -aG "${GROUP_NAME}" "${TARGET_USER}"
fi

# 3) Install udev rules (USB + video4linux + hidraw)
UDEV_RULES_PATH="/etc/udev/rules.d/99-orbbec.rules"
echo ">>> Writing udev rules to ${UDEV_RULES_PATH} ..."
cat > "${UDEV_RULES_PATH}" <<'EOF'
# Orbbec (VID=2bc5): assign devices to the 'orbbec' group with rw for owner+group
# USB parent (helps propagate)
SUBSYSTEM=="usb", ATTR{idVendor}=="2bc5", GROUP:="orbbec", MODE:="0660"

# Video nodes (UVC)
KERNEL=="video[0-9]*", SUBSYSTEM=="video4linux", ATTRS{idVendor}=="2bc5", GROUP:="orbbec", MODE:="0660"

# HID control interface (many Orbbec models expose hidraw for control/sync)
KERNEL=="hidraw*", SUBSYSTEM=="hidraw", ATTRS{idVendor}=="2bc5", GROUP:="orbbec", MODE:="0660"
EOF

# 4) Reload and trigger udev
echo ">>> Reloading udev rules and triggering ..."
udevadm control --reload-rules
udevadm trigger || true

# 5) Make current plugged devices usable immediately (no replug)
echo ">>> Applying immediate permissions to currently plugged devices ..."
apply_immediate() {
  local node="$1"
  # Query udev properties to check vendor id
  if udevadm info -q property -n "$node" 2>/dev/null | grep -qE '^ID_VENDOR_ID=2bc5$'; then
    chgrp orbbec "$node" || true
    chmod g+rw "$node" || true
    echo "    + granted to ${node}"
  fi
}

# video nodes
for v in /dev/video*; do
  [[ -e "$v" ]] || continue
  apply_immediate "$v"
done

# hidraw nodes
for h in /dev/hidraw*; do
  [[ -e "$h" ]] || continue
  apply_immediate "$h"
done

echo ">>> Done."

echo
echo "NEXT STEPS:"
echo "  1) Open a new terminal OR run:  newgrp ${GROUP_NAME}"
echo "     (this refreshes your group membership without reboot)"
echo "  2) Verify:"
echo "       ls -l /dev/video* /dev/hidraw* | egrep 'orbbec|video'"
echo "       python - <<'PY'\nfrom pyorbbecsdk import Context; dl=Context().query_device_list(); print('device count:', dl.get_count() if dl else 0)\nPY"
echo
echo "If you're in a headless container, remember to pass devices into the container:"
echo "  --device=/dev/bus/usb:/dev/bus/usb -v /dev:/dev --group-add ${GROUP_NAME}"

