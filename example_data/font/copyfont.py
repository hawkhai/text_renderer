#encoding=utf8
import os, sys
workspace = r"D:\kSource\pdfai\pdfai_serv\font\workspace"
if not os.path.exists(workspace):
    workspace = r"I:\pdfai_serv\font\workspace"
assert os.path.exists(workspace), "请设置正确的字体识别文件夹"
sys.path.append(workspace)
from pythonx.funclib import *

sys.path.append(r"I:\pdfai_serv\font\data")
from font_config import FONT_CFGS_DATA

def main():
    for name, path, blod, italic, chinese, flags, family, bold_tuning_required in FONT_CFGS_DATA:
        print(path.lower())
        
        fromfile = os.path.join(r"I:\pdfai_serv\font\data\myfonts", path)
        copyfile(fromfile, path.lower())

if __name__ == "__main__":
    main()
    print("ok")
