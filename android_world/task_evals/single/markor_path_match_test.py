import re
s = "/storage/emulated/0/Documents/Markor/1ZLZOLI6Xh/eE9FJKcpPi/zicdJ1kd5I"

target = re.search(r'Markor/.*', s).group(0)

print(target)