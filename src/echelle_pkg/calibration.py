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
            繝?繝ｼ繧ｿ繝輔か繝ｫ繝縺ｮ繝代せ
        files_cmos : dict
            菴ｿ逕ｨ縺吶ｋ繝輔ぃ繧､繝ｫ蜷阪ｒ縺ｾ縺ｨ繧√◆霎樊嶌
        spec : str
            繧ｹ繝壹け繝医Ν繧ｿ繧､繝?
        crop, crop2 : list
            繧ｯ繝ｭ繝?繝礼ｯ?蝗ｲ
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
            豕｢髟ｷ譬｡豁｣繝輔ぃ繧､繝ｫ縺ｮ繝代せ
        orders : list
            蜃ｦ逅?縺励◆縺?繧ｪ繝ｼ繝繝ｼ逡ｪ蜿ｷ
        fit_order_func : function
            轤ｹ謨ｰ縺ｫ蠢懊§縺溷､夐??蠑乗ｬ｡謨ｰ繧定ｿ斐☆髢｢謨ｰ
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

class SensitivityCalibration:
    def __init__(self, path, files_cmos,
                 spec='fujii',
                 crop=[100,1850],
                 crop2=[20,1095],
                 degree=6):
        """
        path : str
            データフォルダのパス
        files_cmos : dict
            キャリブレーションに使用するファイル情報
        spec : str
            スペクトルタイプ
        crop, crop2 : list
            クロップ範囲
        degree : int
            感度フィッティング多項式次数
        """
        self.path = path
        self.files_cmos = files_cmos
        self.spec = spec
        self.crop = crop
        self.crop2 = crop2
        self.degree = degree

        self.cb = None
        self.sphere = None
        self.sphere_back = None

        self.poly_func = None
        self.coefficients = None

        # 積分球の分光放射輝度．黒澤さんの修論P.19(2021)を参考．
        self.wave = np.array(
            [300,310,320,330,340,350,400,450,500,555,600,655,700]
        )
        self.sen = np.array(
            [1.85e4,2.8e4,4.17e4,5.89e4,8.08e4,1.09e5,
             3.67e5,8.03e5,1.37e6,2.05e6,2.56e6,3.10e6,3.42e6]
        )

    # -----------------------------
    # ① Calibrationセットアップ
    # -----------------------------
    def setup_calibration(self):
        self.cb = Calibrations(self.path, self.files_cmos,
                               spec=self.spec,
                               crop=self.crop,
                               crop2=self.crop2)
        self.cb.load_pattern()
        self.cb.load_sphere()
        self.cb.make_cutting_masks()

    # -----------------------------
    # ② sphere画像セットアップ
    # -----------------------------
    def setup_sphere(self, sphere_file, sphere_back_file):
        if self.cb is None:
            raise ValueError("Run setup_calibration() first.")

        self.sphere = EchelleImage(
            os.path.join(self.path, sphere_file),
            clbr=self.cb,
            spec=self.spec,
            crop=self.crop,
            crop2=self.crop2
        )

        self.sphere_back = EchelleImage(
            os.path.join(self.path, sphere_back_file),
            clbr=self.cb,
            spec=self.spec,
            crop=self.crop,
            crop2=self.crop2
        )

        self.sphere.calculate_order_spectra()
        self.sphere_back.calculate_order_spectra()

    # -----------------------------
    # ③ 感度フィット
    # -----------------------------
    def fit_sensitivity(self):
        self.coefficients = np.polyfit(
            self.wave,
            self.sen,
            self.degree
        )
        self.poly_func = np.poly1d(self.coefficients)
        return self.poly_func

    # -----------------------------
    # ④ 感度係数計算
    # -----------------------------
    def calculate_sensitivity(self, cb_ylist,
                              orders=None,
                              start_order=10):

        if self.poly_func is None:
            raise ValueError("Run fit_sensitivity() first.")
        if self.sphere is None:
            raise ValueError("Run setup_sphere() first.")

        if orders is None:
            orders = list(range(len(cb_ylist)))

        coef_sen = []

        for i, nord in enumerate(orders):
            wave_array = cb_ylist[i]
            sen_sampled = self.poly_func(wave_array)

            order_index = start_order + i

            numerator = sen_sampled
            denominator = (
                self.sphere.order_spectra[0, order_index]
                - self.sphere_back.order_spectra[0, order_index]
            )

            coef_sen.append(numerator / denominator)

        return coef_sen
    
    def plot_sensitivity(self, cb_ylist, coef_sen,
                     orders=None,
                     xlabel="Wavelength (nm)",
                     ylabel="Sensitivity Coefficient"):

        if orders is None:
            orders = list(range(len(coef_sen)))

        plt.figure()
        clrs = plt.cm.viridis(np.linspace(0,1,len(coef_sen)))

        for i, nord in enumerate(orders):
            plt.plot(cb_ylist[i], coef_sen[i],
                 c=clrs[i],
                 label=nord)

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid()
        plt.show()