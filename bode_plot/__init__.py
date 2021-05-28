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

import numpy as np

class bode_plot(rtl,spice,thesdk):
    @property
    def _classfile(self):
        return os.path.dirname(os.path.realpath(__file__)) + "/"+__name__

    def __init__(self,*arg): 
        """ Inverter parameters and attributes
            Parameters
            ----------
                *arg : 
                If any arguments are defined, the first one should be the parent instance

            Attributes
            ----------
            proplist : array_like
                List of strings containing the names of attributes whose values are to be copied 
                from the parent

            Rs : float
                Sampling rate [Hz] of which the input values are assumed to change. Default: 100.0e6

            vdd : float
                Supply voltage [V] for bode_plot analog simulation. Default 1.0.

            IOS : Bundle
                Members of this bundle are the IO's of the entity. See documentation of thsdk package.
                Default members defined as

                self.IOS.Members['A']=IO() # Pointer for input data
                self.IOS.Members['Z']= IO() # pointer for oputput data
                self.IOS.Members['control_write']= IO() # Piter for control IO for rtl simulations

            model : string
                Default 'py' for Python. See documentation of thsdk package for more details.
        
            par : boolean
            Attribute to control parallel execution. HOw this is done is up to designer.
            Default False

            queue : array_like
            List for return values in parallel processing. This list is read by the process in parent to get the values 
            evalueted by the instance copies created during the parallel processing loop.

        """
        self.print_log(type='I', msg='Initializing %s' %(__name__)) 
        self.proplist = [ 'Rs' ];    # Properties that can be propagated from parent
        self.Rs =  100e6;            # Sampling frequency
        self.vdd = 1.0
        self.IOS=Bundle()
        self.IOS.Members['A']=IO() # Pointer for input data
        self.IOS.Members['Z']= IO()
        self.IOS.Members['control_write']= IO() 
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

    def main(self):
        ''' The main python description of the operation. Contents fully up to designer, however, the 
        IO's should be handled bu following this guideline:
        
        To isolate the internal processing from IO connection assigments, 
        The procedure to follow is
        1) Assign input data from input to local variable
        2) Do the processing
        3) Assign local variable to output

        '''
        inval=self.IOS.Members['A'].Data
        out=np.array(1-inval)
        if self.par:
            self.queue.put(out)
        self.IOS.Members['Z'].Data=out

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
        if len(arg)>0:
            self.par=True      #flag for parallel processing
            self.queue=arg[0]  #multiprocessing.queue as the first argument
        if self.model=='py':
            self.main()
        else: 
          if self.model=='sv':
              # Verilog simulation options here
              _=rtl_iofile(self, name='A', dir='in', iotype='sample', ionames=['A'], datatype='sint') # IO file for input A
              _=rtl_iofile(self, name='Z', dir='out', iotype='sample', ionames=['Z'], datatype='sint')
              self.rtlparameters=dict([ ('g_Rs',self.Rs),]) #Defines the sample rate
              self.run_rtl()
              self.IOS.Members['Z'].Data=self.IOS.Members['Z'].Data.astype(int)
          if self.model=='vhdl':
              # VHDL simulation options here
              _=rtl_iofile(self, name='A', dir='in', iotype='sample', ionames=['A']) # IO file for input A
              _=rtl_iofile(self, name='Z', dir='out', iotype='sample', ionames=['Z'], datatype='int')
              self.rtlparameters=dict([ ('g_Rs',self.Rs),]) #Defines the sample rate
              self.run_rtl()
              self.IOS.Members['Z'].Data=self.IOS.Members['Z'].Data.astype(int)
          
          elif self.model=='eldo' or self.model=='spectre':
              _=spice_iofile(self, name='A', dir='in', iotype='sample', ionames=['A'], rs=self.Rs, \
                vhi=self.vdd, trise=1/(self.Rs*4), tfall=1/(self.Rs*4))
              _=spice_iofile(self, name='Z', dir='out', iotype='event', sourcetype='V', ionames=['Z'])

              # Saving the analog waveform of the input as well
              self.IOS.Members['A_OUT']= IO()
              _=spice_iofile(self, name='A_OUT', dir='out', iotype='event', sourcetype='V', ionames=['A'])
              #self.preserve_iofiles = True
              #self.preserve_spicefiles = True
              #self.interactive_spice = True
              self.nproc = 1
              self.spiceoptions = {
                          'eps': '1e-6'
                      }
              self.spiceparameters = {
                          'exampleparam': '0'
                      }
              
              # Plotting nodes for interactive eldo purposes.
              # Spectre also supported, but without 'v()' specifiers.
              # i.e. self.plotlist = ['A','Z']
              if self.model == 'eldo':
                  self.plotlist = ['v(A)','v(Z)']

              # Defining library options
              # Path to model libraries needs to be defined in TheSDK.config as
              # either ELDOLIBFILE or SPECTRELIBFILE. In this case, no model libraries
              # will be included (assuming these variables are not defined). The
              # temperature will be set regardless.
              self.spicecorner = {
                          'corner': 'top_tt',
                          'temp': 27,
                      }

              # Example of defining supplies (not used here because the example bode_plot has no supplies)
              #_=spice_dcsource(self,name='dd',value=self.vdd,pos='VDD',neg='VSS',extract=True,ext_start=2e-9)
              #_=spice_dcsource(self,name='ss',value=0,pos='VSS',neg='0')

              # Simulation command
              _=spice_simcmd(self,sim='tran')
              self.run_spice()

          if self.par:
              self.queue.put(self.IOS.Members[Z].Data)

    def define_io_conditions(self):
        '''This overloads the method called by run_rtl method. It defines the read/write conditions for the files

        '''
        # Input A is read to verilog simulation after 'initdone' is set to 1 by controller
        self.iofile_bundle.Members['A'].verilog_io_condition='initdone'
        # Output is read to verilog simulation when all of the outputs are valid, 
        # and after 'initdone' is set to 1 by controller
        self.iofile_bundle.Members['Z'].verilog_io_condition_append(cond='&& initdone')


if __name__=="__main__":
    import matplotlib.pyplot as plt
    from  bode_plot import *
    from  bode_plot.controller import controller as bode_plot_controller
    import pdb
    length=1024
    rs=100e6
    indata=np.random.randint(2,size=length).reshape(-1,1);
    #indata=np.random.randint(2,size=length)
    controller=bode_plot_controller()
    controller.Rs=rs
    #controller.reset()
    #controller.step_time()
    controller.start_datafeed()

    models=[ 'py', 'sv', 'vhdl', 'eldo', 'spectre' ]
    duts=[]
    for model in models:
        d=bode_plot()
        duts.append(d) 
        d.model=model
        d.Rs=rs
        #d.preserve_iofiles=True
        #d.interactive_rtl=True
        #d.interactive_spice=True
        d.IOS.Members['A'].Data=indata
        d.IOS.Members['control_write']=controller.IOS.Members['control_write']
        d.init()
        d.run()

    # Obs the latencies may be different
    latency=[ 0 , 1, 1, 0 ]
    for k in range(len(duts)):
        hfont = {'fontname':'Sans'}
        if duts[k].model == 'eldo' or duts[k].model=='spectre':
            figure,axes = plt.subplots(2,1,sharex=True)
            axes[0].plot(duts[k].IOS.Members['A_OUT'].Data[:,0],duts[k].IOS.Members['A_OUT'].Data[:,1],label='Input')
            axes[1].plot(duts[k].IOS.Members['Z'].Data[:,0],duts[k].IOS.Members['Z'].Data[:,1],label='Output')
            axes[0].set_ylabel('Input', **hfont,fontsize=18);
            axes[1].set_ylabel('Output', **hfont,fontsize=18);
            axes[1].set_xlabel('Time (s)', **hfont,fontsize=18);
            axes[0].set_xlim(0,11/rs)
            axes[1].set_xlim(0,11/rs)
            axes[0].grid(True)
            axes[1].grid(True)
        else:
            figure,axes=plt.subplots(2,1,sharex=True)
            x = np.linspace(0,10,11).reshape(-1,1)
            axes[0].stem(x,indata[0:11,0])
            axes[0].set_ylim(0, 1.1);
            axes[0].set_xlim((np.amin(x), np.amax(x)));
            axes[0].set_ylabel('Input', **hfont,fontsize=18);
            axes[0].grid(True)
            axes[1].stem(x, duts[k].IOS.Members['Z'].Data[0+latency[k]:11+latency[k],0])
            axes[1].set_ylim(0, 1.1);
            axes[1].set_xlim((np.amin(x), np.amax(x)));
            axes[1].set_ylabel('Output', **hfont,fontsize=18);
            axes[1].set_xlabel('Sample (n)', **hfont,fontsize=18);
            axes[1].grid(True)
        titlestr = "Inverter model %s" %(duts[k].model) 
        plt.suptitle(titlestr,fontsize=20);
        plt.grid(True);
        printstr="./inv_%s.eps" %(duts[k].model)
        plt.show(block=False);
        figure.savefig(printstr, format='eps', dpi=300);
    input()
