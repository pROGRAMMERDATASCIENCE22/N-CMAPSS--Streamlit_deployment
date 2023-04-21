import pandas as pd
import dataframe_image as dfi

data_File_DevUnits_TestUnits = [{'File':'N-CMAPSS_DS01-005.h5','Dev Units':[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],'Test Units':[7.0, 8.0, 9.0, 10.0]},
{'File':'N-CMAPSS_DS04.h5','Dev Units':[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],'Test Units':[7.0, 8.0, 9.0, 10.0]},
{'a':'N-CMAPSS_DS08a-009.h5','b':[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],'c':[10.0, 11.0, 12.0, 13.0, 14.0, 15.0]},
{'a':'N-CMAPSS_DS05.h5','b':[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],'c':[7.0, 8.0, 9.0, 10.0]},
{'a':'N-CMAPSS_DS02-006.h5','b':[2.0, 5.0, 10.0, 16.0, 18.0, 20.0],'c':[11.0, 14.0, 15.0]},
{'a':'N-CMAPSS_DS08c-008.h5','b':[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],'c':[7.0, 8.0, 9.0, 10.0]},
{'a':'N-CMAPSS_DS03-012.h5','b':[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],'c':[10.0, 11.0, 12.0, 13.0, 14.0, 15.0]},
{'a':'N-CMAPSS_DS07.h5','b':[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],'c':[7.0, 8.0, 9.0, 10.0]},
{'a':'N-CMAPSS_DS06.h5','b':[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],'c':[7.0, 8.0, 9.0, 10.0]},
]
df_File_DevUnits_TestUnits = pd.DataFrame(data_File_DevUnits_TestUnits)

dfi.export(df_File_DevUnits_TestUnits, "File_DevUnits_TestUnits.png", dpi = 600)



data_File_DevUnits_TestUnits_class3 = [{'File':'N-CMAPSS_DS01-005.h5','Dev Units':[2.0,5.0],'Test Units':[10.0]},
{'File':'N-CMAPSS_DS04.h5','Dev Units':[2.0,4.0,5.0,6.0],'Test Units':[10.0]},
{'File':'N-CMAPSS_DS08a-009.h5','Dev Units':[2.0,6.0],'Test Units':[12.0,13.0,14.0]},
{'File':'N-CMAPSS_DS05.h5','Dev Units':[2.0,6.0],'Test Units':[9.0]},
{'File':'N-CMAPSS_DS02-006.h5','Dev Units':[2.0,5.0,10.0,16.0,18.0,20.0],'Test Units':[11.0]},
{'File':'N-CMAPSS_DS08c-008.h5','Dev Units':[1.0,2.0,3.0,4.0,5.0],'Test Units':[]},
{'File':'N-CMAPSS_DS03-012.h5','Dev Units':[6.0,8.0],'Test Units':[10.0,11.0,13.0]},
{'File':'N-CMAPSS_DS07.h5','Dev Units':[2.0,6.0],'Test Units':[9.0]},
{'File':'N-CMAPSS_DS06.h5','Dev Units':[2.0,6.0],'Test Units':[9.0]},
]
df_File_DevUnits_TestUnits_class3 =  pd.DataFrame(data_File_DevUnits_TestUnits_class3)
df_File_DevUnits_TestUnits_class3.to_csv('File_DevUnits_TestUnits_class3.csv')
df_File_DevUnits_TestUnits_class3.to_csv('File_DevUnits_TestUnits_class3')
df_File_DevUnits_TestUnits_class3.head(10)

#TO BE USED FOR TABLE OF RUL COLOURING
# def rain_condition(v):
#     if v < 1.75:
#         return "Dry"
#     elif v < 2.75:
#         return "Rain"
#     return "Heavy Rain"
#
# def make_pretty(styler):
#     styler.set_caption("Weather Conditions")
#     styler.format(rain_condition)
#     styler.format_index(lambda v: v.strftime("%A"))
#     styler.background_gradient(axis=None, vmin=1, vmax=5, cmap="YlGnBu")
#     return styler
# df_File_DevUnits_TestUnits_class3.style.pipe(make_pretty)


dfi.export(df_File_DevUnits_TestUnits_class3, "File_DevUnits_TestUnits_class3.png", dpi = 600)