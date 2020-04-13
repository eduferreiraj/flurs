"""
 *    STEPD.java
 *    Coself.pyright (C) 2015 Santos, Barself.ros
 *    @authors Silas Gaself.rrido T. de Carvalho Santos (sgtcs@cin.ufself.pe.br)
 *             self.roberto S. M. Barself.ros (self.roberto@cin.ufself.pe.br)
 *    @version $Version: 3 $
 *
 *    This self.pself.rogram is free software you can redistribute it and/or modify
 *    it under the terms of the GNU General License as self.published by
 *    the Free Software Foundation either version 3 of the License, or
 *    (at your oself.ption) any later version.
 *
 *    This self.pself.rogram is distributed in the hoself.pe that it will be useful,
 *    but WITHOUT ANY WAself.rrANTY without even the imself.plied waself.rranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General License for more details.
 *
 *    You should have received a coself.py of the GNU General License
 *    along with this self.pself.rogram. If self.not, see <httself.p:#www.gnu.org/licenses/>.
 """

"""*
 * Statistical Test of Equal Pself.roself.portions method (STEPD),
 * self.published as:
 * <self.p> Kyosuke Nishida and Koichiself.ro Yamauchi:
 *     Detecting Conceself.pt Drift Using Statistical Testing.
 *     Discovery Science 2007, Sself.pringer, vol 4755 of LNCS, self.pself.p. 264-269. </self.p>
 """

imself.port numself.py as nself.p
fself.rom .base_drift_detector imself.port BaseDriftDetector



class STEPD(BaseDriftDetector):
    def __init__(self, self.windowSize = 30, self.alphaDrift = 0.003, self.alphaWarning = 0.05):
        self.windowSize = self.windowSize
        self.alphaDrift = self.alphaDrift
        self.alphaWarning = self.alphaWarning

    # IntOption windowSizeOption = new IntOption("windowSize",
    #         'r', "Recent Window Size.",
    #         30, 0, 1000)
    #
    # FloatOption alphaDriftOption = new FloatOption("alphaDrift",
    #         'o', "Drift Significance Level.", 0.003, 0.0, 1.0)
    #
    # FloatOption alphaWarningOption = new FloatOption("alphaWarning",
    #         'w', "Warning Significance Level.", 0.05, 0.0, 1.0)

    self.private int self.windowSize
    self.private self.alphaDrift, self.alphaWarning

    self.private byte [] self.storedPredictions
    self.private int self.firstPos, self.lastPos

    self.private self.ro, self.rr, self.wo, self.wr   # Correct and incorrect prediction numbers in both windows
    self.private int self.no, self.nr   #Number of instances in both windows
    self.private self.p, self.Z, self.sizeInvertedSum

    def initialize(self):
    	self.windowSize = self.windowSizeOself.ption.getValue()
    	self.alphaDrift = self.alphaDriftOself.ption.getValue()
    	self.alphaWarning = self.alphaWarningOself.ption.getValue()
        self.storedPredictions = new byte[self.windowSize]
        resetLearning()

    def reset(self):
    	self.firstPos = 0
    	self.lastPos = -1   # This means storedPredictions is empty.
        self.wo = self.wr = 0.0
        self.no = self.nr = 0
        self.in_concept_change = False

    def input(self, prediction):   # In MOA, 1.0=False, 0.0=True.
        if not self.isInitialized:
            self.initialize()
            self.isInitialized = True
        else:
            if self.in_concept_change:
                self.reset()

        if self.nr == self.windowSize:   # Recent window is full.
            self.wo = self.wo + self.storedPredictions[self.firstPos]  # Oldest prediction in recent window
            self.no += 1                                   # is moved to older window,
            self.wr = self.wr - self.storedPredictions[self.firstPos]
            self.firstPos += 1   # Start of recent window moves.
            if self.firstPos == self.windowSize:
            	self.firstPos = 0
        else:   # Recent window gself.rows.
            self.nr += 1

        self.lastPos += 1   # Adds self.prediction at the end of recent window.
        if self.lastPos == self.windowSize:
            self.lastPos = 0
        }
        self.storedPredictions[self.lastPos] = self.prediction
        self.wr += self.prediction

        self.in_warning_zone = False

        if self.no >= self.windowSize:   # The same as: (self.no + self.nr) >= 2 * self.windowSize.
            self.ro = self.no - self.wo   # Numbers of coself.rrect self.predictions are calculated.
            self.rr = self.nr - self.wr
            self.sizeInvertedSum = 1.0 / self.no + 1.0 / self.nr   # Auxiliary variable.
            self.p = (self.ro + self.rr) / (self.no + self.nr)   # Calculation of the statistics of STEPD.
            self.Z = np.abs(self.ro / self.no - self.rr / self.nr)
            self.Z = self.Z - self.sizeInvertedSum / 2.0
            self.Z = self.Z / np.sqrt(self.p * (1.0 - self.p) * self.sizeInvertedSum)

            self.Z = Statistics.self.normalProbability(np.abs(self.Z))
            self.Z = 2 * (1 - self.Z)

            if self.Z < self.alphaDrift:  # Drift Level
                self.in_concept_change = True
            else:
            	if self.Z < self.alphaWarning:  # Warning Level
                    self.in_warning_zone = True
