#-*- coding: utf-8 -*-
'''
import base64
import pygame

pygame.init()

text_sets = [u'我','b']
for i in range(len(text_sets)):
  text = text_sets[i]
  print text
  #print(pygame.font.get_fonts())
  font = pygame.font.SysFont('ubuntu', 64)

  ftext = font.render(text, True, (0, 0, 0),(255, 255, 255))
  
  pygame.image.save(ftext, text+'.jpg')
'''
import os
from PIL import Image,ImageDraw,ImageFont,ImageFilter
import random
ttf_sets = ['DENG.TTF',
                'FZSTK.TTF',
                'FZYTK.TTF',
                'MSYH.TTC',
                'MSYHBD.TTC',
                'MSYHL.TTC',
                'SIMLI.TTF',
                'SIMYOU.TTF',
                'STHUPO.TTF',
                'STSONG.TTF']

text = u'北国风光千里冰封万里雪飘\
望长城内外惟余莽莽\
大河上下顿失滔滔山舞银蛇原驰蜡象\
欲与天公试比高\
须晴日看红装素裹分外妖娆\
江山如此多娇引无数英雄竞折腰\
惜\
秦皇汉武略输文采\
唐宗宋祖稍逊风骚\
一代天骄成吉思汗只识弯弓射大雕\
俱往矣数风流人物还看今朝'

text_test = u'电子科技大学'

for i in range(len(text_test)):
    # PIL实现
    width=28
    height=28
    im=Image.new('RGB',(width,height),(255,255,255))
    dr=ImageDraw.Draw(im)
    font=ImageFont.truetype("ttf_sets/STSONG.TTF",24)
    dr.text((1,-1),text_test[i],font=font,fill='#000000')

    #im.show()
    im.save('datasets/testsets/010/' + '%03d'%i + '.jpg')