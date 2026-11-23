alish_qiymeti = input("alish qiymetini gir: ")
satish_qiymeti = input("satish qiymetini gir: ")

if (int(satish_qiymeti) - int(alish_qiymeti))<0:
    print("loss, ziyanin da ", (int(satish_qiymeti) - int(alish_qiymeti)), " manatdi")
elif (int(satish_qiymeti) - int(alish_qiymeti))>0:
     print("gain, ", ((int(satish_qiymeti) - int(alish_qiymeti)), "manat qazandin"))
else:
    print("hecne qazanmadin")
