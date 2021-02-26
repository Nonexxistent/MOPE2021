import random as r
from prettytable import PrettyTable

a0,a1,a2,a3 = 2,4,6,8
X1,X2,X3 = [r.randint(0,20) for i in range(0,8)],[r.randint(0,20) for i in range(0,8)],[r.randint(0,20) for i in range(0,8)]
Y = [(a0+a1*X1[i]+a2*X2[i]+a3*X3[i]) for i in range(0,8)]
X0=[((max(X1)+min(X1))/2),((max(X2)+min(X2))/2),((max(X3)+min(X3))/2)]
dx=[(max(X1)-X0[0]),(max(X2)-X0[1]),(max(X3)-X0[2])]
XN1,XN2,XN3 = [round((X1[i]-X0[0])/dx[0],2) for i in range(0,8)],[round((X2[i]-X0[1])/dx[1],2) for i in range(0,8)],[round((X3[i]-X0[2])/dx[2],2) for i in range(0,8)]
yET=a0+a1*X0[0]+a2*X0[1]+a3*X0[2]
minY=min(Y)
miniy=Y.index(minY)
tp=[X1[miniy],X2[miniy],X3[miniy]]
fkv=a0+a1*X1[miniy]+a2*X2[miniy]+a3*X3[miniy]
table = PrettyTable()
table.field_names = ["#", "X1", "X2", "X3", "Y", "", "XN1", "XN2", "XN3"]
table.add_rows([
        [ "1" ,  X1[0] ,  X2[0] ,  X3[0] , Y[0], "", XN1[0] , XN2[0] , XN3[0] ],
        [ "2" ,  X1[1] ,  X2[1] ,  X3[1] , Y[1], "", XN1[1] , XN2[1] , XN3[1] ],
        [ "3" ,  X1[2] ,  X2[2] ,  X3[2] , Y[2], "", XN1[2] , XN2[2] , XN3[2] ],
        [ "4" ,  X1[3] ,  X2[3] ,  X3[3] , Y[3], "", XN1[3] , XN2[3] , XN3[3] ],
        [ "5" ,  X1[4] ,  X2[4] ,  X3[4] , Y[4], "", XN1[4] , XN2[4] , XN3[4] ],
        [ "6" ,  X1[5] ,  X2[5] ,  X3[5] , Y[5], "", XN1[5] , XN2[5] , XN3[5] ],
        [ "7" ,  X1[6] ,  X2[6] ,  X3[6] , Y[6], "", XN1[6] , XN2[6] , XN3[6] ],
        [ "8" ,  X1[7] ,  X2[7] ,  X3[7] , Y[7], "", XN1[7] , XN2[7] , XN3[7] ],
        [ "X0",  X0[0] ,  X0[1] ,  X0[2] , yET , "",   ""   ,   ""   ,   ""   ],
        [ "dx",  dx[0] ,  dx[1] ,  dx[2] ,  "" , "",   ""   ,   ""   ,   ""   ]])
print(table)
print("Довільні коефіцієнти: a0={0}, a1={1}, a2={2}, a3={3}".format(a0, a1, a2, a3))
print("Рівняння регресії: 2 + 4*X1 + 6*X2 + 8*X3")
print("Еталонне значення Y: 2 + 4*X01 + 6*X02 + 8*X03 =",yET)
print("Оптимальна точка плану (min(Y)): {0} = Y({1}, {2}, {3})".format(fkv,X1[miniy],X2[miniy],X3[miniy]))
