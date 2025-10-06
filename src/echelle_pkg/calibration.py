import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from echelle_pkg.echelle_spectra import Calibrations, EchelleImage

rcParams['ytick.direction'] = 'out'
rcParams['xtick.direction'] = 'out'

class WavelengthCalibration:
    def __init__(self, path, files_cmos, spec='fujii', crop=[100,1850], crop2=[20,1095]):
        """
        path : str
            データフォルダのパス
        files_cmos : dict
            使用するファイル名をまとめた辞書
        spec : str
            スペクトルタイプ
        crop, crop2 : list
            クロップ範囲
        """
        self.path = path
        self.files_cmos = files_cmos
        self.spec = spec
        self.crop = crop
        self.crop2 = crop2
        
        self.cb = None
        self.im = None
        
    def setup_calibration(self):
        self.cb = Calibrations(self.path, self.files_cmos, spec=self.spec,
                               crop=self.crop, crop2=self.crop2)
        self.cb.load_pattern()
        self.cb.load_sphere()
        self.cb.make_cutting_masks()
    
    def setup_image(self, tif_file):
        self.im = EchelleImage(os.path.join(self.path, tif_file), clbr=self.cb,
                               spec=self.spec, crop=self.crop, crop2=self.crop2)
    
    def plot_cut_image(self, idx=0, aspect=6, norm='liner'):
        if self.im is None:
            raise ValueError("Image is not set. Run setup_image() first.")
        self.im.plot_cut_image(idx, aspect=aspect, norm=norm)
    
    def plot_frame(self, idx=0, pattern=True, dark=True):
        if self.im is None:
            raise ValueError("Image is not set. Run setup_image() first.")
        self.im.plot_frame(idx, pattern=pattern, dark=dark)
    
    def calculate_order_spectra(self):
        if self.im is None:
            raise ValueError("Image is not set. Run setup_image() first.")
        self.im.calculate_order_spectra()
        return self.im.order_spectra
    
    def fit_wavelength(self, wcal_file, orders=None, fit_order_func=lambda n: 1 if n<3 else 2):
        """
        wcal_file : str
            波長校正ファイルのパス
        orders : list
            処理したいオーダー番号
        fit_order_func : function
            点数に応じた多項式次数を返す関数
        """
        if orders is None:
            orders = list(range(10, 24))
        
        wcal_path = os.path.join(self.path, wcal_file)
        wcal = pd.read_csv(wcal_path, sep=',', comment='#',
                           names=['ord','from','to','center','wavelength','band'])
        
        wfits = {}
        cb_xlist = []
        cb_ylist = []
        
        plt.figure()
        clrs = plt.cm.viridis(np.linspace(0,1,len(orders)))
        ax = plt.gca()
        
        for j, nord in enumerate(orders):
            p = wcal[wcal['ord']==nord]['center']
            w = wcal[wcal['ord']==nord]['wavelength']
            f = np.poly1d(np.polyfit(p, w, fit_order_func(len(p))))
            wfits[nord] = f
            x = np.arange(self.cb.DIMW)
            cb_xlist.append(x)
            cb_ylist.append(f(x))
            plt.plot(x, f(x), c=clrs[j], label=nord)
            plt.plot(p, w, 'o', c=clrs[j])
            ax.text(-20, f(0), nord, ha='right', va='center')
        
        plt.xlabel('pixel')
        plt.ylabel('wavelength, nm')
        plt.show()
        
        return wfits, cb_xlist, cb_ylist
