import pandas_access as mdb
import pandas as pd
import json
import sys
import pymongo

my_client = pymongo.MongoClient('mongodb+srv://smohan7:Googlesux_2001@cluster0-gxh6r.mongodb.net/test?retryWrites=true&w=majority')
my_database = my_client.test
my_collection = my_database.vdot_pilot2

site_number = 10007
Line = 'Ramp'
Direction = 'South' 
years = ['2015','2016','2017','2018','2019']
key_maint=str(site_number)+'_'+Line+'_'+Direction
type_code_dict={0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'J'}

# mine weather data
df_wea = pd.read_csv("./Bristol_segments_weather_traffic.csv")

df_wea_grp=df_wea.groupby(['Direction'])
df_wea_grp_dir=df_wea_grp.get_group(Direction)
df_wea_grp2=df_wea_grp_dir.groupby(['Type'])
df_wea_grp_type=df_wea_grp2.get_group('Mainline')
df_wea_grp3=df_wea_grp_type.groupby(['SiteNumber'])

try:
	df_wea_grp_final=df_wea_grp3.get_group(int(site_number))
except:
	print("Site number not found in the trf/wea file")
	exit(0)

#print(df_insp_grp_final)

df_wea_data={}
for year in years:
	wea_data=df_wea_grp_final[['{0}_TMAX'.format(year),'{0}_TMIN'.format(year),'{0}_DWT32'.format(year),'{0}_DWT80'.format(year),'{0}_DSNW'.format(year),'{0}_EMSD'.format(year),'{0}_EMXP'.format(year),'{0}_PRCP'.format(year),'{0}_SNOW'.format(year),'{0}_DWTMXN30'.format(year),'{0}_TMAXMIN'.format(year)]]
	df_wea_data[year]=wea_data.to_dict(orient='records')

# mine traffic data
df_trf = pd.read_csv("./Bristol_Ramp_Traffic.csv")

df_trf_grp=df_trf.groupby(['Direction'])
df_trf_grp_dir=df_trf_grp.get_group(Direction)
df_trf_grp2=df_trf_grp_dir.groupby(['Type'])
df_trf_grp_type=df_trf_grp2.get_group(Line)
df_trf_grp3=df_trf_grp_type.groupby(['SiteNumber'])

try:
	df_trf_grp_final=df_trf_grp3.get_group(int(site_number))
except:
	print("Site number not found in the trf/wea file")
	exit(0)



df_trf_data=df_trf_grp_final[['ADT','AAWDT','ADT_4','ADT_BU','ADT_1','ADT_2','ADT_3','ADT_TR','TrafTag']]
yearTraf=df_trf_grp_final.iloc[0]['YearTraf']

# mine inspection data
df_insp = pd.read_csv("./Bristol_Ramp_Asset_TrafficTag.csv")

df_insp_grp=df_insp.groupby(['Direction'])
df_insp_grp_dir=df_insp_grp.get_group(Direction)
df_insp_grp2=df_insp_grp_dir.groupby(['Type'])
df_insp_grp_type=df_insp_grp2.get_group(Line)
df_insp_grp3=df_insp_grp_type.groupby(['SiteNumber'])

try:
	df_insp_grp_final=df_insp_grp3.get_group(int(site_number))
except:
	print("Site number not found in the trf/wea file")
	exit(0)

df_seg_feat = df_insp_grp_final[['Route','MM','Latitude','Longitude']]

KEY=Direction+'_'+str(site_number)+'_'+Line


#print(df_wea_data)

trf_data=json.loads(df_trf_data.reset_index().to_json(orient='index'))
#wea_data=json.loads(df_wea_data.reset_index().to_json(orient='index'))
seg_feat=json.loads(df_seg_feat.reset_index().to_json(orient='index'))

wea_data_final={}
trf_data_final={}
seg_feat_final={}
for key in df_wea_data.keys():
    year_data=df_wea_data[key]
    year_dict={}
    for wea_feats in year_data[0].keys():
        new_key=wea_feats.split('_')[1]
        val=year_data[0][wea_feats]
        year_dict[new_key]=val
    wea_data_final[key]=year_dict

for key in trf_data['0']:
    if key == 'index':
        continue

    trf_data_final[key]=trf_data['0'][key]

for key in seg_feat['0']:
    if key == 'index':
        continue

    seg_feat_final[key]=seg_feat['0'][key]    

print(yearTraf)

json_obj={'_id':KEY,'weather':wea_data_final,'traffic':[{str(yearTraf):trf_data_final}],'segment_features':seg_feat_final}

my_collection.insert(json_obj)



asset_info=pd.read_excel("./AssetCode.xlsx")
asset_info=asset_info[['CODE','MRP DESCRIPTION']]


df_grouped_assets=df_insp_grp_final.groupby(['AssetItemN'])
asset_cnt=0

import pickle
fp='/home/shrey/vdot_db/implement/segment_dictionary_ramp_v2.txt'
with open(fp, 'rb') as f:
    segmentDic = pickle.load(f)

#print(segmentDic[key_maint])
maint_dict=segmentDic[key_maint]
for asset_rec in df_grouped_assets.groups.keys():
        #if asset_cnt>3:
            #break
        asset_data=df_grouped_assets.get_group(asset_rec)
        asset_order=0
        asset_code=asset_info.loc[asset_info['MRP DESCRIPTION']==asset_rec,'CODE'].iloc[0]
       # print(asset_code)
        asset_cnt+=1
        main_data=maint_dict[asset_rec]
        print(asset_rec)
        #break
        maintenance_dict={}
        for year in main_data.keys():
            main_data_yearly=main_data[year]['Maintenance']
            maintenance_dict[str(year)]= main_data_yearly
        #print(maintenance_dict)
        #break
        asset_feat_dict={}
        asset_insp_dict={}
        
        for row in asset_data.itertuples():
            #print(row.Index)
            #asset_feat=row[['Latitude','Longitude','Dimension']]
            #sinspt_feat=row[['Rate_1','Rate_2','Rate_3','Rate_4','Rate_5','Rate_6']]
            #print(asset_feat)
            asset_type_code=str(asset_code)+'_'+type_code_dict[asset_order]
            asset_name=asset_rec+str(asset_order)
            #asset_features=asset_feat.to_frame()
            #asset_feat=pd.DataFrame(asset_feat)
            #asset_feat.insert(0,'Asset_Type_Code',asset_type_code)
            assetFeat_data={'Latitude':row.Latitude,'Longitude':row.Longitude,'Dimension':row.Dimension,'TrafTag':row.TrafTag}
            assetInsp_data={'Rate_1':row.Rate_1,'Rate_2':row.Rate_2,'Rate_3':row.Rate_3,'Rate_4':row.Rate_4,'Rate_5':row.Rate_5,'Rate_6':row.Rate_6} 
            #assetFeat_data=json.loads(asset_feat.reset_index().to_json(orient='index'))
            #assetInsp_data=json.loads(inspt_feat.reset_index().to_json(orient='index'))
            asset_feat_dict[asset_type_code]=assetFeat_data
            asset_insp_dict[asset_type_code]=assetInsp_data
            asset_order+=1
        my_collection.update({'_id':KEY},{'$push':{'Asset_features':{asset_rec:asset_feat_dict}}})
        my_collection.update({'_id':KEY},{'$push':{asset_rec:{'Inspection':{'2016':asset_insp_dict},'Maintenance':maintenance_dict}}})
       
       
          


