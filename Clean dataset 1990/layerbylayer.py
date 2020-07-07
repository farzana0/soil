# layerbylayer.py


response_columns = ['MVA', 'DMP', 'POROTOT1', 'POROTOT3', 'PORODRAI1', 'PORODRAI3', 'CH_cm_h' ]
columns = ['PM3', 'CEC', 'MNM3', 'CUM3'  ,'FEM3' ,'ALM3' ,'BM3'  ,'KM3'  ,'CAM3' ,'MGM3', 'ARGILE', 'SABLE', 'LIMON', 'CentreEp', 'PHSMP', 'PHEAU', 'PCMO']


df1 = pd.read_csv('Couche_Inv1990tot.csv', usecols = ['IDEN2.x'] + ['IDEN3'] + ['GROUPE.x'] + ['Couche'] + columns + response_columns, encoding='latin-1')
df2 = pd.read_csv('Site_Inv1990.csv', usecols = ['IDEN2'] + ['xcoord', 'ycoord'], encoding='latin-1')
df3 = pd.read_csv('Champ_Inv1990.csv', usecols = ['IDEN3', 'Culture_1']  , encoding='latin-1')
print(df1.columns)
print(df2.columns)
print(df3.columns)


# df = pd.merge(df1, df2, left_index=True, right_index=True, how='inner')
df4 = df1.merge(df2, left_on='IDEN2.x', right_on='IDEN2').reindex(columns=['IDEN2', 'IDEN3', 'GROUPE.x', 'Couche' ] + ['xcoord', 'ycoord'] + columns + response_columns)
df = df4.merge(df3, left_on='IDEN3', right_on='IDEN3')


dfcouche1 = df[df['Couche'] == 1]
dfcouche2 = df[df['Couche'] == 2]
dfcouche3 = df[df['Couche'] == 3]

print(dfcouche1.count())
print(dfcouche2.count())
print(dfcouche3.count())





# geopandas_(df)
print(df.columns)