import numpy as np

#Author: Caroline Morley 
def Al2O3(metallicity=0.0):
	logpressure = np.array([-4.51300236407, -4.13238770686, -3.80141843972,-3.45390070922, -3.05673758865, -2.6926713948, -2.2293144208, -1.84869976359, -1.45153664303, -1.02127659574, -0.62411347517, -0.12765957446, 0.352245862884, 0.716312056738, 1.09692671395, 1.39479905437, 1.8085106383 , 2.20567375887 , 2.50354609929 , 2.81796690307 , 3.01654846336 ])
	temp_10_4 = np.array([  6.06053618273,  5.98594967086,  5.89232549929,  5.81764960874,  5.72420419454, 5.64957299334, 5.53740274482, 5.46281623296, 5.38827441044, 5.29491837492,  5.2203765524,  5.12719927425,  5.05288089843,  4.99715328891, 4.92256677705, 4.84775681848, 4.75435609362, 4.64200708773, 4.58610072084, 4.51133545161, 4.45516094867])
	#plt.semilogy(10.0**4/temp_10_4, 10.0**logpressure, linestyle='dashed', color='DarkRed')
	#plt.annotate('Al2O3', xy=(1890, 10**-1), color='DarkRed', rotation=-72,  fontsize='large')#)#, fontsize='small')
	return 10.0**4/temp_10_4, 10.0**logpressure

def Fe(metallicity=0.0): 
	logpressure = np.arange(-6, 3, 0.1)
	temp = 10.0**4/(5.44 - 0.48*logpressure-0.48*metallicity)
	#plt.semilogy(temp, 10.0**logpressure, linestyle='dashed', color='Blue')
	#plt.annotate('Fe', xy=(1750, 10**-0.75), color='Blue', rotation=-60,  fontsize='large')#, fontsize='small')
	return temp, 10.0**logpressure

def Mg2SiO4(metallicity=0.0): 
	logpressure = np.arange(-6, 3, 0.1)
	temp = 10.0**4/(5.89 - 0.37*logpressure-0.73*metallicity)
	#plt.semilogy(temp, 10.0**logpressure, linestyle='dashed', color='Green')
	#plt.annotate('Mg2SiO4', xy=(1560, 10**-1.25), color='Green', rotation=-68,  fontsize='large')#, fontsize='small')
	return temp, 10.0**logpressure

def MgSiO3(metallicity=0.0): 
	logpressure = np.arange(-6, 3, 0.1)
	temp = 10.0**4/(6.26 - 0.35*logpressure-0.70*metallicity)
	#plt.semilogy(temp, 10.0**logpressure, linestyle='dashed', color='Red')
	#plt.annotate('MgSiO3', xy=(1400, 10**-2.25), color='Red', rotation=-72,  fontsize='large')#, fontsize='small')
	return temp, 10.0**logpressure

def Cr(metallicity=0.0): 
	logpressure = np.arange(-6, 3, 0.1)
	temp = 10.0**4/(6.528 - 0.491*logpressure-0.491*metallicity)
	#plt.semilogy(temp, 10.0**logpressure, linestyle='dashed', color='Black')	
	#plt.annotate('Cr', xy=(1330, 10**-2), color='Black', rotation=-63,  fontsize='large')#, fontsize='small')
	return temp, 10.0**logpressure

def MnS(figurenum, metallicity=0.0): 
	logpressure = np.arange(-6, 3, 0.1)
	temp = 10.0**4/(7.45 - 0.42*logpressure-0.84*metallicity)
	#plt.semilogy(temp, 10.0**logpressure, linestyle='dashed', color='Teal')
	#plt.annotate('MnS', xy=(1180, 10**-2.5), color='Teal', rotation=-73,  fontsize='large')#, fontsize='small')
	return temp, 10.0**logpressure

def Na2S(figurenum, metallicity=0.0): 
	logpressure = np.arange(-6, 3, 0.1)
	temp = 10.0**4/(10.05 - 0.72*logpressure-1.08*metallicity)
	#plt.semilogy(temp, 10.0**logpressure, linestyle='dashed', color='DarkOrchid')
	#plt.annotate('Na2S', xy=(835, 10**-2.5), color='DarkOrchid', rotation=-75,  fontsize='large')#, fontsize='small')
	return temp, 10.0**logpressure

def ZnS(figurenum, metallicity=0.0): 
	logpressure = np.arange(-6, 3, 0.1)
	temp = 10.0**4/(12.52 - 0.63*logpressure-1.26*metallicity)
	#plt.semilogy(temp, 10.0**logpressure, linestyle='dashed',color='LimeGreen')
	#plt.annotate('ZnS', xy=(725, 10**-2.25), color='LimeGreen', rotation=-85,  fontsize='large')#, fontsize='small')
	return temp, 10.0**logpressure


def KCl(figurenum, metallicity=0.0): 
	logpressure = np.arange(-6, 3, 0.1)
	temp = 10.0**4/(12.48 - 0.8786*logpressure-0.8786*metallicity)
	#plt.semilogy(temp, 10.0**logpressure, linestyle='dashed', color='MediumVioletRed')	
	#plt.annotate('KCl', xy=(640, 10**-2.25), color='MediumVioletRed', rotation=-85,  fontsize='large')#, fontsize='small')
	return temp, 10.0**logpressure

def NH4H2PO4(figurenum, metallicity=0.0):
	logpressure = np.arange(-6, 3, 0.1)
	temp = 10.0**4/(29.99-0.20*(11.0*logpressure+ 15.0*metallicity))
	#plt.semilogy(temp, 10.0**logpressure, linestyle='dashed', color='Orange')
	#plt.annotate('NH4H2PO4', xy=(290, 10**-2.25), color='Orange', rotation=-85,  fontsize='large')#, fontsize='small')
	return temp, 10.0**logpressure


def H2O(figurenum, metallicity=0.0):
	logpressure = np.array([-3.25228519196,  -2.91224862888, -2.42961608775, -1.87020109689, -1.33272394881, -0.784277879342, -0.26873857404, 0.180987202925, 0.564899451554, 1.0365630713, 1.31078610603, 1.58500914077, 1.89213893967,  2.14442413163, 2.39670932358, 2.68190127971, 3 ])
	temp = [190.789473684, 197.368421053, 203.947368421, 213.815789474,  223.684210526, 233.552631579, 243.421052632, 253.289473684, 263.157894737,  276.315789474, 286.184210526,   292.763157895, 302.631578947, 312.5,  322.368421053, 335.526315789,  348.684210526]
	#plt.semilogy(temp, 10.0**logpressure, linestyle='dashed', color='DodgerBlue')	
	#plt.annotate('H2O', xy=(240, 10**0.0), color='DodgerBlue', rotation=-85,  fontsize='large')#, fontsize='small')
	return temp, 10.0**logpressure

def NH3(figurenum, metallicity=0.0):
	logpressure = np.array([-3.24131627057, -2.95612431444, -2.56124314442 , -2.19926873857 , -1.56307129799, -0.959780621572, -0.422303473492, 
		-0.016453382084, 0.466179159049, 0.937842778793, 1.2449725777, 1.57404021938, 1.8592321755, 2.14442413163, 2.42961608775, 2.72577696527, 3.0])
	temp = 	[111.842105263, 115.131578947, 118.421052632, 121.710526316, 128.289473684,  134.868421053,  141.447368421,   144.736842105, 151.315789474, 
		161.184210526, 164.473684211, 171.052631579, 174.342105263, 180.921052632, 187.5,  194.078947368,  200.657894737]
	#plt.semilogy(temp, 10.0**logpressure, linestyle='dashed', color='DeepPink')	
	#plt.annotate('NH3', xy=(145, 10**1.0), color='DeepPink', rotation=-85,  fontsize='large')#, fontsize='small')
	return temp, 10.0**logpressure


def plotH2OVaporPressure():
	pvap = np.logspace(-6,2,num=100)
	tpvap = 2629.43 / (13.41465 - np.log10(pvap))
	#plot(tpvap, log10(pvap), lw=2, linestyle='dotted', color='Teal')
	return tpvap, np.log10(pvap)

def plotH2OVaporPressure():
	pvap = np.logspace(-8,5,num=100)
	pvap_dynes = pvap * 1e6
	tpvap = 2629.43 / (13.41465 - log10(pvap_dynes))
	#plot(tpvap, log10(pvap/7.54e-4) , lw=2, linestyle='dotted', color='Teal')
	#plot(tpvap, log10(pvap) , lw=2, linestyle='dotted', color='DeepPink')
	return tpvap, np.log10(pvap)


