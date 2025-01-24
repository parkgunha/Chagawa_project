import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn

# df 변수에 담기
audi = pd.read_csv("C:/Users/a/OneDrive/바탕 화면/개인/02. 데이터분석_Bootcamp/006. 머신러닝/04. 머신러닝 프로젝트/1. 데이터_중고차_시세/audi.csv")
bmw = pd.read_csv("C:/Users/a/OneDrive/바탕 화면/개인/02. 데이터분석_Bootcamp/006. 머신러닝/04. 머신러닝 프로젝트/1. 데이터_중고차_시세/bmw.csv")
cclass = pd.read_csv("C:/Users/a/OneDrive/바탕 화면/개인/02. 데이터분석_Bootcamp/006. 머신러닝/04. 머신러닝 프로젝트/1. 데이터_중고차_시세/cclass.csv")
focus = pd.read_csv("C:/Users/a/OneDrive/바탕 화면/개인/02. 데이터분석_Bootcamp/006. 머신러닝/04. 머신러닝 프로젝트/1. 데이터_중고차_시세/focus.csv")
ford = pd.read_csv("C:/Users/a/OneDrive/바탕 화면/개인/02. 데이터분석_Bootcamp/006. 머신러닝/04. 머신러닝 프로젝트/1. 데이터_중고차_시세/ford.csv")
hyundi = pd.read_csv("C:/Users/a/OneDrive/바탕 화면/개인/02. 데이터분석_Bootcamp/006. 머신러닝/04. 머신러닝 프로젝트/1. 데이터_중고차_시세/hyundi.csv")
merc = pd.read_csv("C:/Users/a/OneDrive/바탕 화면/개인/02. 데이터분석_Bootcamp/006. 머신러닝/04. 머신러닝 프로젝트/1. 데이터_중고차_시세/merc.csv")
skoda = pd.read_csv("C:/Users/a/OneDrive/바탕 화면/개인/02. 데이터분석_Bootcamp/006. 머신러닝/04. 머신러닝 프로젝트/1. 데이터_중고차_시세/skoda.csv")
toyota = pd.read_csv("C:/Users/a/OneDrive/바탕 화면/개인/02. 데이터분석_Bootcamp/006. 머신러닝/04. 머신러닝 프로젝트/1. 데이터_중고차_시세/toyota.csv")
unclean_cclass = pd.read_csv("C:/Users/a/OneDrive/바탕 화면/개인/02. 데이터분석_Bootcamp/006. 머신러닝/04. 머신러닝 프로젝트/1. 데이터_중고차_시세/unclean cclass.csv")
unclean_focus = pd.read_csv("C:/Users/a/OneDrive/바탕 화면/개인/02. 데이터분석_Bootcamp/006. 머신러닝/04. 머신러닝 프로젝트/1. 데이터_중고차_시세/unclean focus.csv")
vauxhall = pd.read_csv("C:/Users/a/OneDrive/바탕 화면/개인/02. 데이터분석_Bootcamp/006. 머신러닝/04. 머신러닝 프로젝트/1. 데이터_중고차_시세/vauxhall.csv")
vw = pd.read_csv("C:/Users/a/OneDrive/바탕 화면/개인/02. 데이터분석_Bootcamp/006. 머신러닝/04. 머신러닝 프로젝트/1. 데이터_중고차_시세/vw.csv")

# DF 확인: unclean_cclass, unclean_focus에 다른 df에는 없는 reference 열 확인함
audi.head()
hyundi.head()