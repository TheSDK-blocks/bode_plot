"""
========
Bode plot
========

Bode plot module for TheSystemDevelopmentKit (https://github.com/TheSystemDevelopmentKit). 

Plots frequency response based on given freq, amplitude data.


Initially written by Santeri Porrasmaa, santeri.porrasmaa@aalto.fi

Last modified by Santeri Porrasmaa, 13.09.2021

"""

import os
import sys
if not (os.path.abspath('../../thesdk') in sys.path):
    sys.path.append(os.path.abspath('../../thesdk'))

from thesdk import *
from rtl import *
from spice import *
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import MultipleLocator, AutoMinorLocator, LogLocator
from matplotlib.offsetbox import AnchoredText
import numpy as np

class bode_plot(thesdk):
    @property
    def _classfile(self):
        return os.path.dirname(os.path.realpath(__file__)) + "/"+__name__

    def __init__(self,*arg): 
        """ bode plot parameters and attributes
            Parameters
            ----------
                *arg : 
                If any arguments are defined, the first one should be the parent instance

            Attributes
            ----------
            proplist : array_like
                List of strings containing the names of attributes whose values are to be copied 
                from the parent

            IOS : Bundle
                Members: 'in', constains Numpy 2-d array of frequency, complex voltage points.
                Used to calculate transfer function.
            IOS.Members['vin'] : 2D-np.array
                Input voltage data, containing frequency and voltage data columns (freq assumed as first column)

            IOS.Members['vout'] : 2D-np.array
                Output voltage data, containing frequency and voltage data columns (freq assumed as first column)
            model : string
                Default 'py' for Python. 
            calc_tf : bool
                Internally controlled variable. DO NOT TOUCH!
                If both vin and vout are given, calculate transfer function as vout/vin and plot result
            tf : 2-D np.array
                If transfer function was calculated, store in this variable. First column is frequency,
                second is transfer function.
            squared : bool
                If true, plot squared magnitude response
                Default: false
            freq : np.array
                frequency data vector
            cutoff_level : float
                This value is substracted from signal to find cutoff.
                E.g. to find -6 dB cutoff, give value as 6.
                Default: 3. Given relative to maximum.
            cutoff : List[float]
                List containing detected cut-off frequencies
            gain_margin : float
                Difference between 180 degree point and UGF
            phase_margin : float
                180+phase value at UGF
            resp_type: string
                Expected response type: "LP" | "HP" | "BP"
                Used to find cutoffs
            annotate_cutoff: bool
                Annotate cutoffs in figure?
            add_cutoff_line : bool
                Add a line in the figure at cutoff point?
            annotate_margins: bool
                Annotate amplitude and phase margins in figure?
            mag_plot: bool
                Plot magnitude data
            mag_label: string
                Label of y-axis for magnitude plots
            phase_plot: bool
                Plot phase data
            phase_label: string
                Label omagnitude f y-axis for phase plots
            shade_area: bool
                If true, shade the area under the TF curve and y == 0
                Limits for shading set with self.fstart and self.fstop.
                Useful for annotating e.g. bandwidth of input signal, etc
            fstart: float
                Frequency from which shading starts
            fstop: float
                Frequency to which shading stops
            plot_title: string
                Main title of generated plot
            plot: bool
                Genereate figures if true
            degrees: bool
                If true, plot phase response in degrees rather than radians.
                Default: true
            dc_gain: float
                Property storing the DC gain of the transfer function (in dB).
            xlim: Tuple(float, float)
                If given, sets plot x-axis limits according to this
            xscale: string
                'log' | 'lin'
                Sets x-axis tick spacing to linear or logarithmic
            figformat : string
                'pdf' | 'eps' | 'svg' | 'png' | etc.. (see Matplotlib savefig())
                Format to save figure in
            save_fig: bool
                If true, saves figure to path specified by save_path
            save_path: str
                Path to save the figure to. File extension is set from figformat.
                Default: '../figures/bode_plot.<figformat>'
            reuse_fig: bool
                Plot to current active figure window. Useful for
                plotting multiple TF's in same figure. Currently
                supported for either mag_data or phase_data only
                (not both).
        """
        self.print_log(type='I', msg='Initializing %s' %(__name__)) 
        self.proplist = [  ];    # Properties that can be propagated from parent
        self.IOS=Bundle()
        self.IOS.Members['vin']=IO() # Pointer for input data
        self.IOS.Members['vout']=IO()
        self.figformat='pdf' # Format for image file, default: '.pdf'. This is automatically appended to the end of self.save_path
        self.calc_tf=False # internally controlled variable
        self.freq=None # Frequency axis data used in plotting
        self.cutoff = [] # Cutoff freq vector
        self.tf=None # The transfer function is stored in this variable
        self.cutoff_level=3 # The level from which the cutoff frequency is to be calculated from (default -3dB)
        self.dc_gain=0 # DC gain
        self.mag_plot=True # Draw magnitude plot?
        self.phase_plot=True # Draw phase plot?
        self.squared=False # If true, plot squared magnitude response
        self.annotate_cutoff=True # Annotate cut-off frequency?
        self.add_cutoff_line=True # Add a line to cutoff?
        self.annotate_margins=True # Annotate amplitude and phase margins?
        self.mag_label=None # Label for magnitude plot
        self.phase_label=None # Label for phase plot
        self.shade_area=False # True to shade area under curve between fstart and fstop
        self.fstart=None # Freq from which to start shading
        self.fstop=None # Freq to which shading stops
        self.plot_title = '' # Main title for figure window
        self.plot = True # True to plot the reponse
        self.resp_type = 'LP' # Expected response type: "LP" | "HP" | "BP" currently supported
        self.degrees=True# Calculate argument of transfer function in degrees?
        self.xlim=None
        self.xscale='log' # Scale for plot x-axis
        self.save_fig=False
        self.save_path='../figures/bode_plot'
        self.reuse_fig=False # Plot current TF to active figure window? Useful for plotting multiple TF's in same figure

        # Table of order of magnitudes v. SI prefixes
        self.si_table = {
                12:'T',
                9:'G',
                6:'M',
                3:'k',
                0:'',
                -3:'m',
                -6:'u',
                -9:'n',
                -12:'p',
                -15:'f',
                }

        # File for control is created in controller
        self.model='py';             # Can be set externally, but is not propagated
        self.par= False              # By default, no parallel processing
        self.queue= []               # By default, no parallel processing

        # this copies the parameter values from the parent based on self.proplist
        if len(arg)>=1:
            parent=arg[0]
            self.copy_propval(parent,self.proplist)
            self.parent =parent;

        self.init()

    def init(self):
        """ Method to re-initialize the structure if the attribute values are changed after creation.

        """
        pass #Currently nohing to add

    def get_si_prefix_str(self, val):
        exp=np.log10(val)
        for key, val in self.si_table.items():
            if exp >= key:
                break
        return key, val 

    def format_si_str(self, val, prec=3):
        exp, prefix = self.get_si_prefix_str(val)
        retstr='%' + ('.%d' % prec) + 'g '
        retstr=(retstr % (val/(10**exp))) + prefix 
        return retstr

    def get_cutoff(self, magdata):
        cutoff_level = self.cutoff_level
        max_val=max(magdata)
        if cutoff_level < 0:
            self.print_log(type='W', msg='Cut-off level given as negative! Converting to positive!')
            self.cutoff_level = -1 * self.cutoff_level
            cutoff_level=self.cutoff_level
        cutoff_level = max_val - cutoff_level # Cut-off should be relative to maximum value 
        arr=np.abs(magdata-cutoff_level)
        if self.resp_type.lower()=='lp' or self.resp_type.lower()=='hp':
            idx1=arr.argmin()
            return [self.freq[idx1]]
        elif self.resp_type.lower()=='bp':
            idx1 = arr.argmin()
            arr=np.delete(arr, idx1)
            idx2 = arr.argmin()
            return [self.freq[idx1], self.freq[idx2]]
        else:
            self.print_log(type='F', msg='Unsupported response type %s' % self.resp_type)

    def get_margins(self, magdata,phasedata):
        # Where the phase is closest to +-180 degrees
        phase_pos=np.argmin(np.abs(np.abs(phasedata)-180))
        # Where the amplitude is closest to 0dB
        amp_pos=np.argmin(np.abs(magdata))
        # Unity gain frequency (where amp is 0dB)
        self.ugf=self.freq[amp_pos]
        # Compute phase margin (180+phasedata)
        self.phase_margin=np.round(180+phasedata[amp_pos],1)
        # Compute gain margin (difference between -180 degree point to UGF)
        self.gain_margin=np.round(magdata[amp_pos]-magdata[phase_pos],1)
        self.print_log(type='I',msg=f"The phase margin is {self.phase_margin}")
        self.print_log(type='I',msg=f"The gain margin is {self.gain_margin}")


    def check_input(self):
        '''
        Check that the input is of correct format. Expected is 2-by-m matrix where first colmun is freq and second is magnitude.
        '''
        vout_mat=self.IOS.Members['vout'].Data
        vin_mat=self.IOS.Members['vin'].Data
        nrows,ncols=vout_mat.shape
        if nrows==2 and ncols>2:
            vout_mat=vout_mat.transpose() # Make column vectors
        elif ncols==2 and nrows>2:
            pass
        else:
            self.print_log(type='F', msg='The IO vout should contain two columns of data!')
        if isinstance(vin_mat,np.ndarray):
            # If vin is also given, calculate result as transfer function
            self.calc_tf=True
            nrows,ncols=vin_mat.shape
            if nrows==2 and ncols>2:
                vin_mat=vin_mat.transpose() # Male column vectors
            elif ncols==2 and nrows>2:
                pass
            else:
                self.print_log(type='F', msg='The IO vin should containt two columns of data!')
            if len(vin_mat[:,1]) != len(vout_mat[:,1]):
                maxlen = min(max(vin.shape), max(vin.shape))
                self.print_log(type='W', msg='Input and output voltage vectors are not of equal length, clipping both to %d samples' % maxlen)
                vin=vin[0:maxlen]
                vout=vout[0:maxlen]
        else: # Otherwise, plot only vout
            self.calc_tf=False
        self.freq=vout_mat[:,0].real
        if self.calc_tf:
            return np.vstack((vout_mat[:,0], vout_mat[:,1]/vin_mat[:,1])).T
        else:
            return vout_mat

    def shade_curve(self, data):
        '''
        Shades the area between the magnitude/phase plot and y==0 (e.g. assumes positive TF)

        Useful for annotating bandwidth
        '''
        if self.fstart:
            x1=self.fstart
            x1_idx=np.where(self.freq>=x1)[0]
            if len(x1_idx)>0:
                x1_idx=x1_idx[0]
            else:
                self.print_log(type='W', msg='No frequency corresponding to fstart: %.4g detected in input data!' % self.fstart)
                x1_idx=0
        else:
            x1=0
            x1_idx=0
        if self.fstop:
            x2=self.fstop
            x2_idx=np.where(self.freq>=x2)[0]
            if len(x2_idx)>0:
                x2_idx=x2_idx[0]
            else:
                self.print_log(type='W', msg='No frequency corresponding to fstop: %.4g detected in input data!' % self.fstop)
                x2_idx=-1
        else:
            x2=self.freq[-1]
            x2_idx=-1
        x=self.freq[x1_idx:x2_idx] if x2_idx!=-1 else self.freq[x1_idx:] 
        y=data[x1_idx:x2_idx] if x2_idx!= -1 else data[x1_idx:]
        plt.fill_between(x, y, y2=0, alpha=0.2)

    def main(self):
        ''' The main python description of the operation. Contents fully up to designer, however, the 
        IO's should be handled bu following this guideline:
        
        To isolate the internal processing from IO connection assigments, 
        The procedure to follow is
        1) Assign input data from input to local variable
        2) Do the processing
        3) Assign local variable to output

        '''
        keys=sys.modules.keys()
        if 'plot_format' in keys and self.mag_plot and self.phase_plot:
            self.print_log(type='W', msg='plot_format module detected! Using plot_format may provide poor results when plotting mag. and phase to same plot!')
            figsize=(1.1*3.5,1.25*1.8)
            self.print_log(type='W', msg=f'Setting figure size to {figsize}')
            plt.rcParams['figure.figsize'] = figsize
        # Input signal processing, check dimensions are correct, etc..
        # After calling this, transfer function and frequency points are 
        # available from their respective variables (see documentation)
        data=self.check_input()
        # Calculate magnitude and phase from transfer function
        if self.phase_plot:
            if self.squared:
                phase_data=np.angle(data[:,1]**2, deg=self.degrees).real
            else:
                phase_data=np.angle(data[:,1], deg=self.degrees).real
            self.phase_data=phase_data
        if self.squared:
            data[:,1] = 20*np.log10(np.abs(data[:,1])**2)
            self.tf=data
            mag_data=data[:,1].real # to suppress errros while plotting (this still has imag part of 0j)
        else:
            data[:,1] = 20*np.log10(np.abs(data[:,1]))
            self.tf=data
            mag_data=data[:,1].real

        self.dc_gain=mag_data[0]
        self.print_log(type='I', msg='DC gain (extracted at freq. %.2g Hz) is %.2f dB' % (self.freq[0], self.dc_gain))
        # Get cutoff frequency
        self.cutoff=self.get_cutoff(mag_data)
        self.cutoff.sort()
        for f in self.cutoff:
            self.print_log(type='I', msg='Cut-off frequency is: %.4g Hz.' % f)        
        # Get amplitude and phase margins
        self.get_margins(mag_data,phase_data)
        # Update labels, if not already given:
        if self.mag_label is None:
            self.mag_label='Magnitude (dB)'
        if self.phase_label is None:
            if self.degrees:
                self.phase_label='Phase (deg.)'
            else:
                self.phase_label='Phase (rad.)'
        # Set x-axis limits based on data, if not already given
        if not self.xlim:
            self.xlim=(self.freq[0], self.freq[-1])
        if self.xscale == 'lin': # Matplotlib wants it to be 'linear'... 
            self.xscale='linear'
        if self.xscale not in ('linear', 'log'):
            self.print_log(type='F', msg='Unsupported x-axis scale %s!' % self.xscale)
        if self.mag_plot and self.phase_plot: 
            fig, ax = plt.subplots(2,1,sharex=True)
            subfig1=ax[0].plot(self.freq,mag_data)
            ax[0].set_ylabel(self.mag_label) 
            ax[0].set_xscale(self.xscale)
            ax[0].set_xlim(*self.xlim)
            if self.annotate_cutoff:
                ax[0].set_ylim(bottom=min(mag_data)) # This avoids vertical line from strecting the y-limit
                for i,f in enumerate(self.cutoff):
                    if self.add_cutoff_line:
                        ax[0].axvline(x=f, linestyle='--')
                    txt=AnchoredText('$A_{DC}=%3.1f$ dB\n $f_{c,%d}=$ %sHz' % (self.dc_gain,i,self.format_si_str(f)),
                            loc='lower center')
                    ax[0].add_artist(txt)
            subfig2=ax[1].plot(self.freq,phase_data)
            ax[1].set_ylabel(self.phase_label) 
            ax[1].set_xlabel('Frequency (Hz)')
            ax[1].set_xlim(*self.xlim)
            ax[1].set_xscale(self.xscale)
            ax[1].grid(True, which='both') 
            if self.annotate_margins:
                ax[1].set_ylim(bottom=min(phase_data)) # This avoids vertical line from strecting the y-limit
                if self.add_cutoff_line:
                    ax[1].axvline(x=f, linestyle='--')
                if self.degrees:
                    txt=AnchoredText(f'GBW: {self.format_si_str(self.ugf)}Hz\n Gain margin: {self.gain_margin} dB\n Phase margin: {self.phase_margin}$^\circ$' , loc='lower center')
                else:
                    txt=AnchoredText(f'GBW: {self.format_si_str(self.ugf)}Hz\n Gain margin: {self.gain_margin} dB\n Phase margin: {self.phase_margin} rad' , loc='lower center')
                ax[1].add_artist(txt)
            fig.align_ylabels(ax[:])
            if self.plot_title:
                fig.suptitle(self.plot_title)
            if self.plot:
                plt.show(block=False)
                plt.pause(0.5)
        elif self.mag_plot and not self.phase_plot:
            if self.reuse_fig:
                fig=plt.gcf()
            else:
                fig=plt.figure()
            lines=plt.plot(self.freq, mag_data)
            ax=plt.gca()
            ax.set_ylabel(self.mag_label)
            ax.set_xlabel('Frequency (Hz)')
            ax.set_xscale(self.xscale)
            ax.set_xlim(*self.xlim) 
            ax.grid(True, which='both')
            if self.plot_title:
                fig.suptitle(self.plot_title)
            if self.shade_area:
                self.shade_curve(mag_data)
            if self.annotate_cutoff:
                ax.set_ylim(bottom=min(mag_data)) # This avoids vertical line from extending the y-limit
                for i,f in enumerate(self.cutoff):
                    ax.axvline(x=f, linestyle=lines[-1].get_linestyle(), color=lines[-1].get_color())
                    txt=AnchoredText('$f_{c,%d}=$%sHz' % (i,self.format_si_str(f)), loc='lower center') # TODO: figure out a way to automatically determine best position
                    ax.add_artist(txt)
            if self.plot:
                plt.show(block=False)
                plt.pause(0.5)
        elif self.phase_plot and not self.mag_plot:
            if self.reuse_fig:
                fig=plt.gcf()
            else:
                fig=plt.figure()
            plt.plot(self.freq, phase_data)
            ax=plt.gca()
            ax.set_ylabel(self.phase_label)
            ax.set_xlabel('Frequency (Hz)')
            ax.set_xscale(self.xscale)
            ax.set_xlim(*self.xlim) 
            ax.grid(True, which='both')
            if self.plot_title:
                fig.suptitle(self.plot_title)
            if self.shade_area:
                self.shade_curve(phase_data)
            if self.plot:
                plt.show(block=False)
                plt.pause(0.5)
        else:
            self.print_log(type='I', msg='mag_plot and phase_plot flags were false: no plots produced!')
        if self.save_fig and (self.phase_plot or self.mag_plot):
            fig.savefig('%s.%s' % (self.save_path, self.figformat), format=self.figformat)

    def run(self,*arg):
        ''' The default name of the method to be executed. This means: parameters and attributes 
            control what is executed if run method is executed. By this we aim to avoid the need of 
            documenting what is the execution method. It is always self.run. 

            Parameters
            ----------
            *arg :
                The first argument is assumed to be the queue for the parallel processing defined in the parent, 
                and it is assigned to self.queue and self.par is set to True. 
        
        '''
        if self.model=='py':
            self.main()
            if self.par:
                self.queue.put(
                        {**self.IOS.Members}
                        )


if __name__=="__main__":
    import matplotlib.pyplot as plt
    from  bode_plot import *
    from  bode_plot.controller import controller as bode_plot_controller
    import pdb

    models=[ 'py']
    duts=[]
    vout=np.loadtxt('vout.txt',dtype='complex')
    vin=np.loadtxt('vin.txt',dtype='complex')
    gain=100
    vout[:,1]=vout[:,1]*gain
    for model in models:
        d=bode_plot()
        duts.append(d) 
        d.model=model
        d.degrees=True
        d.squared=True
        d.IOS.Members['vout'].Data=vout
        d.IOS.Members['vin'].Data=vin
        d.save_fig=True
        d.save_path='../figures/bode'
        d.mag_plot=True
        d.phase_plot=True
        d.init()
        d.run()

    # Obs the latencies may be different
    input()
