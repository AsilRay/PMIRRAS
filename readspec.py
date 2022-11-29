from typing import OrderedDict
import numpy as np

import time
import os.path
import scipy
from scipy.special import jv
from scipy.interpolate import interp1d
from lmfit import minimize, Parameters, Model

class lvpmi:
	
	def __init__(self,filepath):
		self.filepath   = filepath
		self.filename   = filepath
		self._read_header(filepath)
		
	def _read_header(self,filepath):
		self.f = open(filepath,'rb')
		self.f.seek(0,0)
		self.date = np.frombuffer(self.f.read(6),dtype='uint16')[:]
		self.time = np.frombuffer(self.f.read(8),dtype='uint16')[:]
		self.startNbr = np.frombuffer(self.f.read(2),dtype='uint16')[0]
		self.endNbr = np.frombuffer(self.f.read(2),dtype='uint16')[0]
		self.Res = np.frombuffer(self.f.read(2),dtype='uint16')[0]
		self.Pts = np.frombuffer(self.f.read(2),dtype='uint16')[0]
		
		
		self.f.close()
		return (self.time,self.endNbr,self.Res,self.Pts,self.startNbr) 
		
	def getsize(self):
		# Size in int8 bytes:
		self.f = open(self.filepath,'rb')
		data_wl     = np.frombuffer(self.f.read(), dtype = 'uint8')
		size   = np.shape(data_wl)
		return(size)

	def getNbrSpectra(self):
		self.NbrSpec     =  int(self.getsize()/self.Pts/8) -1
		print('File contains: '+str(self.NbrSpec)+' Spectra.')
		return(self.NbrSpec )

	def getspecdata(self):
		self.f = open(self.filepath,'rb')
		NbrSpec     = self.getNbrSpectra()
		#array = self._read_header(self.filepath)
		data        = np.zeros([(NbrSpec),self.Pts])
		meastime 	= np.zeros([(NbrSpec),4])
		#nu          = np.linspace(array[1], array[2], self.Pts)
		self.f.seek(32,0)
		for i in range(NbrSpec):
			meastime[i,:] = np.frombuffer(self.f.read(8),dtype='uint16')
			data[i,:] = np.frombuffer(self.f.read(int(self.Pts*8)),dtype='float64')
			self.f.seek(32+(i+1)*int(self.Pts*8 +8))
		return(meastime, data)

	def getRegion(self, cut_range):
		print(cut_range)
		multiplier = self.Pts/self.startNbr
		a = int(self.Pts-multiplier*cut_range[0])
		b = int(self.Pts-multiplier*cut_range[1])
		print(a,b)
		x_cut = np.linspace(int(cut_range[0]),int(cut_range[1]), b-a)
		_,R = self.getspecdata()
		R_cut = R[:,a:b]
		
		return x_cut, R_cut
	def scaleSpec(self, R_cut, rangePlot=[4000,1000], method='mean', rangeBG=[2200,1700]):
		nu2Pt = interp1d(np.linspace(rangePlot[0],rangePlot[1],R_cut.shape[1]), np.linspace(1,R_cut.shape[1],R_cut.shape[1]))
		ScaleValue = []
		if method == 'max':
			ScaleValue.append(np.max(R_cut[:,int(nu2Pt(rangePlot[0])):int(nu2Pt(rangePlot[1]))],axis=1))
			ScaleValue = np.squeeze(ScaleValue)
		if method == 'mean':
			for i in range(rangeBG.shape[0]):
				ScaleValue.append(np.mean(R_cut[:,int(nu2Pt(rangeBG[i][0])):int(nu2Pt(rangeBG[i][1]))],axis=1))
			ScaleValue = np.mean(ScaleValue,axis=0)
		ScaleFactor=1/np.array(ScaleValue)
		R_cut_scale = np.zeros([self.NbrSpec, R_cut.shape[1]])
		for i in range(R_cut.shape[0]):
				R_cut_scale[i,:]= R_cut[i,:]*(ScaleFactor[i])
		
		return ScaleValue, ScaleFactor , R_cut_scale
	def besselj21(self,params,x):
			y  = (params[0])*(x)*np.abs(jv(2,params[1]*(x)))
			return y
	def besselj22(self,params,x):
			Ax = (params[1]*x**2 + params[2]*x + params[3])
			y  = params[0]*np.abs(jv(2,Ax))
			return y
	def residual(self,Params, x, data,eps_data, order):
			if order==1:
				a1 = Params['a1'].value
				amp = Params['amp'].value
				model = self.besselj21([amp,a1],x)
			if order ==2:
				a1 = Params['a1'].value
				a2 = Params['a2'].value
				a3 = Params['a3'].value
				amp = Params['amp'].value
				model = self.besselj22([amp,a1,a2,a3],x)
			return (data-model)/eps_data
	def initFitParam(self,order=1, values=[4,0.62]):
		self.fitparams = Parameters()
		if order == 1:
			itemslist = ['amp','a1']
			self.fitparams.add('amp', value=values[0], min=values[0]-1, max=values[0]+1)#, min = 2, max=3)
			self.fitparams.add('a1',value = values[1], min =values[0]-1, max= values[0]+1)
		if order == 2:
			itemslist = ['amp','a1', 'a2','a3']
			self.fitparams.add(itemslist[0], value=values[0],     min=values[0]-50, max=values[0]+50)#, min = 2, max=3)
			self.fitparams.add(itemslist[1], value = values[1], min =-5,   max= 5)
			self.fitparams.add(itemslist[2], value = values[2], min=-5,    max = 5)
			self.fitparams.add(itemslist[3], value=values[3],   min = -5,  max=5)
		return self.fitparams, itemslist

	def BesselFit(self,x, R, params, itemslist, rangeNu, order, eps_data):
		bessel_fits = np.zeros(R.shape)
		for i in range(self.NbrSpec):	
			out = minimize(self.residual, params, args=(x,R[i,:],eps_data, order))
			resid = out.residual
			items=[out.params[j] for j in itemslist]
			itemsValue = [items[j].value for j in range(len(itemslist))]
			if order ==1:
				bessel_fits[i,:] = self.besselj21(itemsValue,x)
			if order == 2:
				bessel_fits[i,:] = self.besselj22(itemsValue,x)
		print('hello')
		return out
	def BGpolyfit(self,R,x,order): # R is the reflectivity data, x = x-axis of the reflectivity data
		R0 = np.zeros(np.shape(R))
		R_corr_cut = np.zeros(np.shape(R))
		for i in range(R0.shape[0]):
				R0[i,:] = np.poly1d(np.polyfit(x, R[i,:], order))(x)
				R_corr_cut[i,:] = (R[i,:]-R0[i,:])/R0[i,:] 
		
		return R_corr_cut, R0
		
	def gaussian(self,x, amp, cen, wid):
		return amp * np.exp(-(x-cen)**2 / wid)
	def Gaussfit(self,raw_corr, cut_range):
		NbrSpec = np.shape(raw_corr)[0]
		gauss_fits = np.zeros([NbrSpec,len(raw_corr[0,:])])
		sigma = np.zeros([NbrSpec,1])
		center = np.zeros([NbrSpec,1])
		peak_area = np.zeros([NbrSpec,1])
		x = np.linspace(-10, 10, len(raw_corr[0,:]))
		nu = np.linspace(cut_range[0],cut_range[1], len(raw_corr[0,:]))
		init_vals = [1, 0, 1]  # for [amp, cen, wid]
		for s in range(NbrSpec-1):
			best_vals, covar = scipy.optimize.curve_fit(self.gaussian, x, raw_corr[s,:], p0=init_vals)
			gauss_fits[s,:] = self.gaussian(x,best_vals[0],best_vals[1],best_vals[2])
			sigma[s] = np.sqrt(best_vals[2]/2)#*(plot_range[1]-plot_range[0])/20
			center[s]= best_vals[1]
			peak_area[s] = np.sum(gauss_fits[s,int(center[s]-2*sigma[s]):int(center[s]-2*sigma[s])])
		return gauss_fits, center, sigma
