# matplotlibwx
PythonのmatplotlibとwxPythonを利用して，散布図，ベクトル線図，カラーコンターをGUI操作で描くPythonスクリプトです．

フォルダごとダウンロードして，matplotlibwx.pyをPythonで実行して下さい．

以下のモジュールをpipでダウンロードして下さい．
- python 3の場合: pip install matplotlib numpy wxpython xlrd==1.2.0 openpyxl pillow urllib3 requests pyperclip
- python 2の場合: pip install matplotlib numpy wxPython==4.1.0 xlrd==1.2.0 openpyxl pillow urllib3 requests pyperclip

英語環境で使えば，英語表記になるはずです．
強制的に英語表記にしたい場合，matplotlibwx.pyをテキストエディタで開き，先頭付近に書いてある#languages = ['en']のコメントを外して保存してから実行して下さい．

---

This is a Python script for drawing scatter plots, vector plots, and color contours using Python's matplotlib and wxPython with GUI operations.

Download the folder and run matplotlibwx.py in Python.

Download the following modules with pip.
- For python 3: pip install matplotlib numpy wxpython xlrd==1.2.0 openpyxl pillow urllib3 requests pyperclip
- For python 2: pip install matplotlib numpy wxPython==4.1.0 xlrd==1.2.0 openpyxl pillow urllib3 requests pyperclip

If you use it in an English environment, it should be written in English.
If you want to force it to be in English, open matplotlibwx.py in a text editor, uncomment #languages = ['en'] near the top, save the file, and run it.
