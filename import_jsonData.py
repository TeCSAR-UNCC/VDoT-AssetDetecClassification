import pandas as pd
import json
import sys
import csv
import math

with open('output_json.json', 'r') as f:
    seg_dict = json.load(f)

#print(seg_dict)


asset_info=pd.read_excel("./AssetCode.xlsx")
asset_info=asset_info[['CODE','MRP DESCRIPTION']]

#asset_info['MRP DESCRIPTION']==asset_rec
asset="Paved Shoulders"
key=seg_dict['_id']
seg_data=seg_dict['segment_features']
keyy=key.split('_')
Direction=keyy[0]
SiteNumber=keyy[1]
Type=keyy[2]
route=seg_data['Route']
mm=seg_data['MM']
lat=seg_data['Latitude']
longt=seg_data['Longitude']
#dimension=seg_data['Dimension']
traffic_data=seg_dict['traffic'][0]
weather_data=seg_dict['weather']
YearTraf=list(traffic_data.keys())[0]
traff_data_dict=traffic_data[YearTraf]
wea_data_dict={}
trf_heads=['ADT','AAWDT','ADT_4','ADT_BU','ADT_1','ADT_2','ADT_3','ADT_TR']
wea_heads=[]
for year in weather_data.keys():
    wea_data=weather_data[year]
    for wea_head in wea_data.keys():
        wea_data_dict['FY'+year+'_'+wea_head]=wea_data[wea_head]
        wea_heads.append('FY'+year+'_'+wea_head)

#print(wea_data_dict)
#print(wea_heads)
#file_headers=[]
asset_names=asset_info['MRP DESCRIPTION'].tolist()
########################################################################
#mine possible maintenance orders for each asset

main_info=pd.read_excel("./asset_maintenance.xlsx")
main_info=main_info.to_dict(orient='records')

asset_mainCode_dict={}
for rec in main_info:
    asset_list=[]

    asset=rec['MRP DESCRIPTION']
    #asset_list.append(asset)
    if math.isnan(rec['CONDITION BASED AND NON-CYCLICAL']) == False:
        asset_list.append(rec['CONDITION BASED AND NON-CYCLICAL'])
    if math.isnan(rec['PREVENTIVE AND CYCLICAL']) == False:
        asset_list.append(rec['PREVENTIVE AND CYCLICAL'])
    if math.isnan(rec['REPAIR/CORRECTIVE']) == False:
        asset_list.append(rec['REPAIR/CORRECTIVE'])
    if math.isnan(rec['RESTORE/REPLACE']) == False:
        asset_list.append(rec['RESTORE/REPLACE'])
    if math.isnan(rec['Others']) == False:
        asset_list.append(rec['Others'])
    for att in rec.keys():
        if 'Unnamed' in att:
            if '7' in str(rec[att]):
                asset_list.append(rec[att]) 
    asset_mainCode_dict[asset]=asset_list

#asset_mainCode_dict has maintenance order codes for each asset as a dictionary
#############################################################################
Asset_features=seg_dict['Asset_features']
#print(Asset_features[0])

for asset in asset_names:
        if asset in seg_dict.keys():
            #asset='Brush & Tree'
            asset_data=seg_dict[asset][0]
            insp_data=asset_data['Inspection']             
            main_data=asset_data['Maintenance']
            #print(main_data)
            #print(asset)
            main_data_dict={}
            main_heads=[]
            for main_years in main_data.keys():
                main_rec=main_data[main_years]
                if main_rec==[]:
                    main_orders=asset_mainCode_dict[asset]
                    for order in main_orders:
                        main_header='FY'+str(main_years)+'_'+str(int(order))
                        main_heads.append(main_header)
                        main_data_dict[main_header]=0
                else:
                    existing_unique_orders=[]
                    for orders in main_rec:
                        if orders[1] not in existing_unique_orders: 
                            existing_unique_orders.append(orders[1])
                    #for unique_order in existing_unique_orders:
                    main_orders=asset_mainCode_dict[asset]  
                    for order in main_orders:
                        if order in existing_unique_orders:
                            main_header='FY'+str(main_years)+'_'+str(int(order))
                            main_heads.append(main_header)
                            main_data_dict[main_header]=1
                        else:
                            main_header='FY'+str(main_years)+'_'+str(int(order))
                            main_heads.append(main_header)
                            main_data_dict[main_header]=0      
            
            for asset_feat in Asset_features:
                if asset in asset_feat.keys():
                    asset_feats=asset_feat[asset]
            print(asset_feats)
            '''
            for year in insp_data.keys():
                #print(year)
                #print(year_data[year])
                ind=0
                asset_df=pd.DataFrame()
                for asset_code in insp_data[year].keys():
                    #print(asset_code)
                    #print(year_data[year][asset_code])
                    AssetItemTag=asset_code
                    AssetID=asset_code.split('_')[0]
                    insp_rec= insp_data[year][asset_code] 
                    file_headers=['AssetItemTag','SiteNumber','Route','MM','Direction','Type','AssetID','AssetItemName','Latitude','Longitude','Rate_1','Rate_2','Rate_3','Rate_4','Rate_5','Rate_6']+trf_heads+wea_heads+main_heads
                    #print(file_headers)
                    mydict={'SiteNumber':SiteNumber,'Route':route,'MM':mm,'Direction':Direction,'Type':Type,'AssetItemName':asset,'Latitude':lat,'Longitude':longt,
                    **insp_rec,**traff_data_dict,**wea_data_dict,**main_data_dict}
                    asset_df=asset_df.append(pd.DataFrame(mydict,index=[ind]))
                
                    asset_df.to_csv('./ArcGIS_files/'+asset+'.csv', header=file_headers)
                    ind+=1
                
                with open(asset+'.csv', 'w') as csv_file:  
                    writer = csv.writer(csv_file)
                    for key, value in mydict.items():
                        writer.writerow([key, value])
                '''
                
'''
with open(asset+'.csv', 'w') as f:  # Just use 'w' mode in 3.x
    w = csv.DictWriter(f, my_dict.keys())
    w.writeheader()
    w.writerow(my_dict)
'''
