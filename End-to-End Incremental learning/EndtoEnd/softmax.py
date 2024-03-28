class SoftmaxDiffLoss(dagnn.ElementWise):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.opts = {}
        self.mode = 'MI'
        self.temperature = 2
        self.origstyle = 'multiclass'
        self.opts_vl = {}
        self.average = 0
        self.numAveraged = 0

    def forward(self, inputs, params):
        outputs = []
        outputs.append(vl_nnsoftmaxdiff(inputs[0], inputs[1], [], self.opts_vl))
        n = self.numAveraged
        m = n + inputs[0].shape[3]
        self.average = (n * self.average + np.array(outputs[0])) / m
        self.numAveraged = m
        return outputs

    def backward(self, inputs, params, derOutputs):
        derInputs = []
        derInputs.append(vl_nnsoftmaxdiff(inputs[0], inputs[1], derOutputs[0], self.opts_vl))
        derInputs.append([])
        derParams = []
        return derInputs, derParams

    def reset(self):
        self.average = 0
        self.numAveraged = 0

    def getOutputSizes(self, inputSizes, paramSizes):
        outputSizes = []
        outputSizes.append([1, 1, 1, inputSizes[0][3]])
        return outputSizes

    def getReceptiveFields(self):
        rfs = []
        rfs.append({'size': [np.nan, np.nan], 'stride': [np.nan, np.nan], 'offset': [np.nan, np.nan]})
        rfs.append(rfs[0])
        return rfs

    def __init__(self, *args, **kwargs):
        self.load(*args, **kwargs)
        self.opts_vl['mode'] = self.mode
        self.opts_vl['temperature'] = self.temperature
        self.opts_vl['origstyle'] = self.origstyle
