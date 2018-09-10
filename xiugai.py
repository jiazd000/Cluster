# -*- coding: utf-8 -*-
import numpy as np
f=open(r'ex2data1.txt','r')

s=f.read()

s=s.replace(',',' ')
a=0
print len(s)

for i in range(len(s)):
    if s[i]==',':

        a=a+1
f.close()
print a
f=open(r'ex2data1.txt','w')
f.write(s)
f.close()


