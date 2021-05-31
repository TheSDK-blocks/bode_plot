"""
========
Inverter
========

Inverter model template The System Development Kit
Used as a template for all TheSyDeKick Entities.

Current docstring documentation style is Numpy
https://numpydoc.readthedocs.io/en/latest/format.html

For reference of the markup syntax
https://docutils.sourceforge.io/docs/user/rst/quickref.html

This text here is to remind you that documentation is iportant.
However, youu may find it out the even the documentation of this 
entity may be outdated and incomplete. Regardless of that, every day 
and in every way we are getting better and better :).

Initially written by Marko Kosunen, marko.kosunen@aalto.fi, 2017.


Role of section 'if __name__=="__main__"'
--------------------------------------------

This section is for self testing and interfacing of this class. The content of it is fully 
up to designer. You may use it for example to test the functionality of the class by calling it as
``pyhon3.6 __init__.py``

or you may define how it handles the arguments passed during the invocation. In this example it is used 
as a complete self test script for all the simulation models defined for the bode_plot. 

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
import numpy as np

class bode_plot(rtl,spice,thesdk):
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

            model : string
                Default 'py' for Python. 
        
        """
        self.print_log(type='I', msg='Initializing %s' %(__name__)) 
        self.proplist = [  ];    # Properties that can be propagated from parent
        self.IOS=Bundle()
        self.IOS.Members['vin']=IO() # Pointer for input data
        self.IOS.Members['vout']=IO()
        self.calc_tf=False # internally controlled variable
        self.freq=None # Frequency axis data used in plotting
        self.tf=None # The transfer function is stored in this variable
        self.mag_plot=True # Draw magnitude plot?
        self.phase_plot=True # Draw phase plot?
        self.annotate_cutoff=True # Annotate cut-off frequency?
        self.mag_label=None # Label for magnitude plot
        self.phase_label=None # Label for phase plot
        self.plot_title = 'Bode plot'
        self.resp_type = 'LP' # Expected response type
        self.degrees=False # Calculate argument of transfer function in degrees?
        self.xlim=None
        self.save_fig=False
        self.save_path=''
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

    def get_cutoff(self, magdata):
        cutoff_level = -6 # TODO: Add this as parameter?
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

    def check_input(self):
        '''
        Check that the input is of correct format.
        '''
        vout_mat=self.IOS.Members['vout'].Data
        vin_mat=self.IOS.Members['vin'].Data
        nrows,ncols=vout_mat.shape
        if nrows==2 and ncols>2:
            vout_mat=vout_mat.transpose()
        elif ncols==2 and nrows>2:
            pass
        else:
            self.print_log(type='F', msg='The IO vout should contain two columns of data!')
        freq=vout_mat[:,0].real
        vout=vout_mat[:,1]
        # If vin is also given, calculate result as transfer function
        if isinstance(vin_mat,np.ndarray):
            self.calc_tf=True
            nrows,ncols=vin_mat.shape
            if nrows==2 and ncols>2:
                vin_mat=vin_mat.transpose()
            elif ncols==2 and nrows>2:
                pass
            else:
                self.print_log(type='F', msg='The IO vin should containt two columns of data!')
            vin=vin_mat[:,1]
            if len(vin) != len(vout):
                maxlen = min(len(vin), len(vout))
                self.print_log(type='W', msg='Input and output voltage vectors are not of equal length, clipping both to %d samples' % maxlen)
                vin=vin[0:maxlen]
                vout=vout[0:maxlen]
                freq=freq[0:maxlen]
        else: # Otherwise, plot only vout
            self.calc_tf=False
        self.freq=freq
        if self.calc_tf:
            self.tf=vout/vin
        else:
            self.tf=vout
        return


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
        # Input signal processing, check dimensions are correct, etc..
        # After calling this, transfer function and frequency points are 
        # available from their respective variables (see documentation)
        self.check_input()
          
        # Calculate magnitude and phase from transfer function
        mag_data=20*np.log10(np.abs(self.tf))
        phase_data=np.angle(self.tf, deg=self.degrees)
        # Get cutoff frequency
        if self.annotate_cutoff:
            cutoff=self.get_cutoff(mag_data)
            for f in cutoff:
                self.print_log(type='I', msg='Cut-off frequency is: %.4g Hz.' % f)        
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
        if self.mag_plot and self.phase_plot: 
            fig, ax = plt.subplots(2,1,sharex=True)
            subfig1=ax[0].plot(self.freq,mag_data)
            ax[0].set_ylabel(self.mag_label) 
            ax[0].set_xscale('log')
            ax[0].set_xlim(*self.xlim)
            if self.annotate_cutoff:
                cutoff=sorted(cutoff)
                ax[0].set_ylim(bottom=min(mag_data)) # This avoids vertical line from strecting the y-limit
                for i,f in enumerate(cutoff):
                    ax[0].axvline(x=f, linestyle='--')
                    plt.text(0.5,0.05+0.2*i, '$f_{c%d}=%.2g$' % (i,f),transform=ax[0].transAxes)
            subfig2=ax[1].plot(self.freq,phase_data)
            ax[1].set_ylabel(self.phase_label) 
            ax[1].set_xlabel('Frequency (Hz)')
            ax[1].set_xlim(*self.xlim)
            ax[1].set_xscale('log')
            ax[1].grid(True) 
            fig.suptitle(self.plot_title)
            plt.show(block=False)
            if self.save_fig:
                plt.savefig(self.save_path,format='pdf')
        elif self.mag_plot and not self.phase_plot:
            fig=plt.plot(self.freq, mag_data)
            ax=plt.gca()
            ax.set_ylabel(self.mag_label)
            ax.set_xlabel('Frequency (Hz)')
            ax.set_xscale('log')
            ax.set_xlim(*self.xlim) 
            if self.annotate_cutoff:
                cutoff=sorted(cutoff)
                ax.set_ylim(bottom=min(mag_data)) # This avoids vertical line from extending the y-limit
                for i,f in enumerate(cutoff):
                    ax.axvline(x=f, linestyle='--')
                    plt.text(0.5,0.05+0.2*i, '$f_{c%d}=%.2g$' % (i,f),transform=ax.transAxes)
            plt.show(block=False)
            if self.save_fig:
                plt.savefig(self.save_path,format='pdf')
        elif self.phase_plot and not self.mag_plot:
            fig=plt.plot(self.freq, phase_data)
            ax=plt.gca()
            ax.set_ylabel(self.phase_label)
            ax.set_xlabel('Frequency (Hz)')
            ax.set_xscale('log')
            ax.set_xlim(*self.xlim) 
            plt.show(block=False)
            if self.save_fig:
                plt.savefig(self.save_path,format='pdf')
        else:
            self.print_log(type='I', msg='mag_plot and phase_plot flags were false: no plots produced!')
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
    for model in models:
        d=bode_plot()
        duts.append(d) 
        d.model=model
        d.degrees=True
        d.IOS.Members['vout'].Data=vout
        d.IOS.Members['vin'].Data=vin
        d.init()
        d.run()

    # Obs the latencies may be different
    input()
