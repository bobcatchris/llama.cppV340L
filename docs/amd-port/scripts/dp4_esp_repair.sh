#!/usr/bin/env bash
exec > /tmp/dp4_esp_repair.log 2>&1
set -x
export HOME=/home/chris
# 1. source ESP pollution check (13A6-0001): look for the clone-UUID cfg
mkdir -p /mnt/src_esp_check
mount UUID=13A6-0001 /mnt/src_esp_check
echo "=== source EFI/BOOT:"
ls /mnt/src_esp_check/EFI/BOOT/ 2>/dev/null
grep -rl "1185036a" /mnt/src_esp_check/EFI/ 2>/dev/null && {
  sed -i 's/1185036a-a604-45a2-8acc-490294bd08ac/35d23f30-9386-4480-ba2f-7caa684b69a3/g' $(grep -rl "1185036a" /mnt/src_esp_check/EFI/ 2>/dev/null)
  echo "source pollution removed"
}
umount /mnt/src_esp_check
rmdir /mnt/src_esp_check

# 2. populate the clone ESP (BF43-1AE1)
mkdir -p /mnt/clone_esp/EFI/BOOT /mnt/clone_esp/EFI/ubuntu
mount --bind /boot/efi /mnt/src_esp_check 2>/dev/null || mount UUID=13A6-0001 /mnt/src_esp_check
rsync -a /mnt/src_esp_check/ /mnt/clone_esp/
umount /mnt/src_esp_check

# 3. stub fix on the clone: search the CLONE root uuid
sed -i 's/35d23f30-9386-4480-ba2f-7caa684b69a3/1185036a-a604-45a2-8acc-490294bd08ac/g' /mnt/clone_esp/EFI/ubuntu/grub.cfg

# 4. fallback cfg for the removable path
cat > /mnt/clone_esp/EFI/BOOT/grub.cfg << 'CFG'
search.fs_uuid 1185036a-a604-45a2-8acc-490294bd08ac root hd0,gpt2
set prefix=($root)'/boot/grub'
configfile $prefix/grub.cfg
CFG

# 5. verify
echo "=== clone ESP files:"
find /mnt/clone_esp -type f | sed 's|/mnt/clone_esp||'
echo "=== stub:"
grep search.fs_uuid /mnt/clone_esp/EFI/ubuntu/grub.cfg
echo "=== fallback:"
cat /mnt/clone_esp/EFI/BOOT/grub.cfg
umount /mnt/clone_esp
echo "REPAIR-DONE"
