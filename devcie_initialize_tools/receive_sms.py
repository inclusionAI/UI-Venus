

#!/usr/bin/env python3
"""
receive_sms.py  ——  向连接的 Android 设备注入一条“收到的短信”

用法:
    python receive_sms.py \
        --device 47.110.238.119:15558 \
        --sender +14379969633 \
        --text "Hello from nowhere"

依赖:
    - 已安装 adb 并能被 PATH 找到
    - 目标设备已 `adb connect IP:PORT` 或 USB 直连，能 `adb shell` 进去
"""
import argparse, base64, datetime, subprocess, sys, textwrap

# ----------  GSM/3GPP 编码工具 ---------- #
def _bcd(n: int) -> int:              # 把十进制 0-99 编成 BCD
    return (n // 10) << 4 | (n % 10)

def _swap_bcd(address: str) -> bytes: # 电话号 -> 交换半字节
    digits = address.lstrip('+')
    if len(digits) % 2:
        digits += 'F'
    return bytes(int(digits[i+1] + digits[i], 16) for i in range(0, len(digits), 2))

def _septet_pack(text: str) -> bytes: # GSM7 bit-packing
    bits = []
    for c in text:
        b = ord(c) & 0x7F
        bits.extend([(b >> i) & 1 for i in range(7)])
    out, carry, cbits = bytearray(), 0, 0
    for bit in bits:
        carry |= bit << cbits
        cbits += 1
        if cbits == 8:
            out.append(carry); carry = 0; cbits = 0
    if cbits:
        out.append(carry)
    return bytes(out)

def build_sms_deliver_pdu(sender: str, message: str) -> str:
    """返回 base64(3GPP PDU)，可直接传给 injectSmsPduForSubscriber"""
    sca = b'\x00'                     # SMSC 地址留空，走系统默认
    first_octet = b'\x04'             # MT SMS-DELIVER
    addr = _swap_bcd(sender)
    addr_hdr = bytes([len(sender.lstrip('+')), 0x91]) + addr
    pid_dcs = b'\x00\x00'
    now = datetime.datetime.utcnow()
    ts = bytes([
        _bcd(now.year % 100), _bcd(now.month), _bcd(now.day),
        _bcd(now.hour), _bcd(now.minute), _bcd(now.second), 0x00
    ])
    user_data = _septet_pack(message)
    pdu = sca + first_octet + addr_hdr + pid_dcs + ts + bytes([len(message)]) + user_data
    return base64.b64encode(pdu).decode()

# ----------  调用 adb 注入 ---------- #
def run(cmd: list[str]) -> subprocess.CompletedProcess:
    subprocess.run(['adb', 'connect', '47.110.238.119:15557'], check=True)
    return subprocess.run(cmd, text=True, capture_output=True, check=False)

def inject_sms(dev: str, pdu_b64: str) -> None:
    """
    Android 6+ 先尝试  cmd phone sms inject …  (走 TelephonyManager API)
    若提示 unknown/permission denied，再降级走 service call isms
    """
    # A) 首选 cmd phone sms inject (Android 8 起自带；部分 6/7 也 back-port 了)
    cmd_inject = [
        "adb", "-s", dev, "shell", "cmd", "phone", "sms",
        "inject", "--format", "3gpp", "--priority", "0", pdu_b64
    ]
    r = run(cmd_inject)
    if r.returncode == 0 and "Exception" not in r.stderr:
        print("✓ 已通过 cmd phone sms inject 注入")
        return
    print("cmd phone sms inject 失败，降级使用 service call isms ...")

    # B) 退而求其次：直接 Binder 调 ISms.injectSmsPduForSubscriber()
    #   不同 Android 版本序号不同，最保险办法是走 ‘cmd phone’ 带 --slot / --subId
    #   若设备不带该 API，只能 root 或放弃
    sub_id = "0"          # 单卡设备一般是 0
    slot_id = "0"         # 单卡一般是 0
    cmd_isms = [
        "adb", "-s", dev, "shell", "service", "call", "isms", "11",
        "i32", sub_id,          # subId
        "i32", slot_id,         # phoneId/slot
        "s16", "null",          # callingPackage
        "s16", "null",          # attributionTag
        "s16", pdu_b64,         # PDU(base64)
        "s16", "3gpp",          # format
        "s16", "null", "s16", "null"  # sentIntent, deliveryIntent
    ]
    r2 = run(cmd_isms)
    if r2.returncode == 0 and "Exception" not in r2.stdout+r2.stderr:
        print("✓ 已通过 service call isms 注入 (API 11)")
    else:
        print("× 仍然失败，请确认设备权限或 Android 版本。\n"
              "stderr:", r2.stderr.strip(), "\nstdout:", r2.stdout.strip())
        sys.exit(1)

# ----------  主程 ---------- #
def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent("""\
            利用 adb 在 Android 真机/云手机上模拟收到短信
            依赖项：adb ≥ 29；目标机须允许 shell 用户调用 injectSmsPdu"""))
    ap.add_argument("--device", required=True, help="adb 设备名或 ip:port")
    ap.add_argument("--sender", required=True, help="发信号码，如 +15551234567")
    ap.add_argument("--text",   required=True, help="短信正文（英数字符即可）")
    args = ap.parse_args()

    pdu_b64 = build_sms_deliver_pdu(args.sender, args.text)
    inject_sms(args.device, pdu_b64)

if __name__ == "__main__":
    main()