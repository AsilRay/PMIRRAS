def loadOpus(folder,file):

    opus_wol = opus(folder+file)
    data_wol = np.frombuffer(opus_wol.f.read(), dtype = 'uint8')

    opus_wol = opus(folder+file)
    data_wol_sgl = np.frombuffer(opus_wol.f.read(), dtype = 'float32')
    
    return data_wol, data_wol_sgl
def getPts(data_4):
    startPts = []
    NbrPts   = []
    for i in range(data_4.shape[0]):
        data_str_4 = "".join([chr(item) for item in data_4[i,:]])
        if 'END' in data_str_4:
            startPts.append(i)
        elif 'NPT' in data_str_4:
            NbrPts.append(data_uint8[i*4 + 8:i*4 + 12].view(dtype='uint32')[0])
    return startPts, NbrPts
def normalize(interfs):
    if len(np.array(interfs).shape)==1:
            interfs = interfs-np.mean(interfs)
    if len(np.array(interfs).shape) >=1:
         for i,ele in enumerate(interfs):
            interfs = interfs[i]-np.mean(interfs[i])
def apodize(interfs, windowfuncname):
    if len(np.array(interfs).shape)==1:
        N = len(interfs)
        window=np.zeros([N])
        interfs_a = []
        mid_wdw=np.argmin(interfs)
        windowfunc = get_window(windowfuncname,2*mid_wdw)
        window[0:mid_wdw] = windowfunc[0:mid_wdw]
        if 2*mid_wdw <= N:
            window[mid_wdw:2*mid_wdw]= windowfunc[mid_wdw:2*mid_wdw]
        elif 2*mid_wdw > N:
            window[mid_wdw:N]= windowfunc[mid_wdw:N]
        interfs_a.append(interfs*window)
    elif len(np.array(interfs).shape) >=1:
        N = len(interfs[0])
        window=np.zeros([N])
        interfs_a = []
        for i,ele in enumerate(interfs):
            mid_wdw=np.argmin(interfs[i])
            windowfunc = get_window(windowfuncname,2*mid_wdw)
            window[0:mid_wdw] = windowfunc[0:mid_wdw]
            if 2*mid_wdw <= N:
                window[mid_wdw:2*mid_wdw]= windowfunc[mid_wdw:2*mid_wdw]
            elif 2*mid_wdw > N:
                window[mid_wdw:N]= windowfunc[mid_wdw:N]
            interfs_a.append(interfs[i]*window)
    return window,interfs_a

def zeroFill(fillF, interfs):
    interfs_zf = []
    for i,ele in enumerate(interfs):
        N = len(interfs[i])
        interf_zf = np.zeros([16384*fillF])
        interf_zf[0:N] = interfs[i][0:N]
        interfs_zf.append(interf_zf)
    return interfs_zf