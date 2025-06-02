import matplotlib.font_manager as fm

# 列出系统中所有字体的文件路径和名称
for font in fm.fontManager.ttflist:
    # 只打印包含“黑体”、“微软雅黑”等关键字的字体
    if "SimHei" in font.name or "Hei" in font.name or "YaHei" in font.name or "Kai" in font.name:
        print(font.name, font.fname)
