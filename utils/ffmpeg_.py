from pyadl import *

def getGPUtype():
    adv = ADLManager.getInstance().getDevices()
    ac = []
    for a in adv:
        ab = [str(a.adapterIndex), str(a.adapterName)]
        ac.append(ab)
    return ac


