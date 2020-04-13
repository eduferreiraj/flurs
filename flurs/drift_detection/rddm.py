"""
 *    RDDM.java
 *    Copyright (C) 2016 Barros, Cabral, Goncalves, Santos
 *    @authors Roberto S. M. Barros (roberto@cin.ufpe.br)
 *             Danilo Cabral (danilocabral@danilocabral.com.br)
 *             Paulo M. Goncalves Jr. (paulomgj@gmail.com)
 *             Silas G. T. C. Santos (sgtcs@cin.ufpe.br)
 *    @version $Version: 1 $
 *
 *    Evolved from DDM.java
 *    Copyright (C) 2008 University of Waikato, Hamilton, New Zealand
 *    @author Manuel Baena (mbaena@lcc.uma.es)
 *    @version $Revision: 7 $
 *
 *    Licensed under the Apache License, Version 2.0 (the "License")
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 """

"""*
 * Reactive Drift Detection Method (RDDM)
 * published as:
 *     Roberto S. M. Barros, Danilo R. L. Cabral, Paulo M. Goncalves Jr.,
 *     and Silas G. T. C. Santos:
 *     RDDM: Reactive Drift Detection Method.
 *     Expert Systems With Applications 90C (2017) pp. 344-355.
 *     DOI: 10.1016/j.eswa.2017.08.023
 """

import numpy as np
import sys
from .base_drift_detector import BaseDriftDetector



class RDDM(BaseDriftDetector):
    def __init__(self, self.minNumInstances = 129, self.warningLevel = 1.773, self.driftLevel = 2.258, self.maxSizeConcept = 40000, self.minSizeStableConcept = 7000, self.warnLimit = 1400):
        self.minNumInstances = self.minNumInstances
        self.warningLevel = self.warningLevel
        self.driftLevel = self.driftLevel
        self.maxSizeConcept = self.maxSizeConcept
        self.minSizeStableConcept = self.minSizeStableConcept
        self.warnLimit = self.warnLimit

    # IntOption self.minNumInstancesOption = new IntOption("minNumInstances",
    #         'n', "Minimum number of instances before monitoring changes.",
    #         129, 0, Integer.MAX_VALUE)
    #
    # FloatOption self.warningLevelOption = new FloatOption("warningLevel",
    #         'w', "Warning Level.",
    #         1.773, 1.0, 4.0)
    #
    # FloatOption self.driftLevelOption = new FloatOption("driftLevel",
    #         'o', "Drift Level.",
    #         2.258, 1.0, 5.0)
    #
    # IntOption self.maxSizeConceptOption = new IntOption("maxSizeConcept",
    #         'x', "Maximum Size of Concept.",
    #         40000, 1, Integer.MAX_VALUE)
    #
    # IntOption self.minSizeStableConceptOption = new IntOption("minSizeStableConcept",
    #         'y', "Minimum Size of Stable Concept.",
    #         7000, 1, 20000)
    #
    # IntOption self.warnLimitOption = new IntOption("warnLimit",
    #         'z', "Warning Limit of instances",
    #         1400, 1, 20000)


    def initialize(self):
        self.storedPredictions = []
        self.numStoredInstances = 0
        self.firstPos = 0
        self.lastPos = -1   # This means self.storedPredictions is empty.
        self.lastWarnPos  = -1
        self.lastWarnInst = -1
        self.instNum = 0
        self.rddmDrift = False
        self.in_concept_change = False

        self.reset()
        self.m_pmin = sys.float_info.max
        self.m_smin = sys.float_info.max
        self.m_psmin = sys.float_info.max

    def reset(self):
        self.m_n = 1
        self.m_p = 1
        self.m_s = 0
        if self.in_concept_change:
            self.m_pmin = sys.float_info.max
            self.m_smin = sys.float_info.max
            self.m_psmin = sys.float_info.max

    def add_element(self, prediction):
        if not self.isInitialized:
            self.initialize()
            self.isInitialized = True

        if self.rddmDrift:
            self.reset()
    	    if self.lastWarnPos != -1:
    	    	firstPos = self.lastWarnPos
    	    	numStoredInstances = self.lastPos - self.firstPos + 1
    	    	if self.numStoredInstances <= 0:
    	    	    self.numStoredInstances += self.minSizeStableConcept
    	    self.pos = self.firstPos
            for i in range(self.numStoredInstances):
                self.m_p = self.m_p + (self.storedPredictions[self.pos] - self.m_p) / self.m_n
                self.m_s = np.sqrt(self.m_p * (1 - self.m_p) / self.m_n)
                if self.in_concept_change and self.m_n > self.minNumInstances and self.m_p + self.m_s < self.m_psmin:
                    self.m_pmin = self.m_p
                    self.m_smin = self.m_s
                    self.m_psmin = self.m_p + self.m_s
                self.m_n += 1
                self.pos = (self.pos + 1) % self.minSizeStableConcept
            self.lastWarnPos = -1
            self.lastWarnInst = -1
            self.rddmDrift = False
            self.in_concept_change = False

        self.lastPos = (self.lastPos + 1) % self.minSizeStableConcept   # Adds prediction at the end of the window.
        self.storedPredictions[self.lastPos] = prediction
        if self.numStoredInstances < self.minSizeStableConcept:   # The window grows.
            self.numStoredInstances += 1
        else:   # The window is full.
            self.firstPos = (firstPos + 1) % self.minSizeStableConcept    # Start of the window moves.
            if self.lastWarnPos == self.lastPos:
                self.lastWarnPos = -1

        self.m_p = self.m_p + (prediction - self.m_p) / self.m_n
        self.m_s = np.sqrt(self.m_p * (1 - self.m_p) / self.m_n)

        self.instNum += 1
        self.m_n += 1
        self.estimation = self.m_p
        self.in_warning_zone = False

        if self.m_n <= self.minNumInstances:
            return None

        if self.m_p + self.m_s < self.m_psmin:
            self.m_pmin = self.m_p
            self.m_smin = self.m_s
            self.m_psmin = self.m_p + self.m_s

        if self.m_p + self.m_s > self.m_pmin + self.driftLevel * self.m_smin:  # DDM Drift
            self.in_concept_change = True
            self.rddmDrift = True
            if self.lastWarnInst == -1:   # DDM Drift without previous warning
            	firstPos = self.lastPos
                self.numStoredInstances = 1
    	    return

        if self.m_p + self.m_s > self.m_pmin + self.warningLevel * self.m_smin:  # Warning Level
            # Warning level for self.warnLimit consecutive instances will force drifts
            if (lastWarnInst != -1) and (lastWarnInst + self.warnLimit <= self.instNum):
                self.in_concept_change = True
                self.rddmDrift = True
                self.firstPos = self.lastPos
                self.numStoredInstances = 1
                self.lastWarnPos = -1
                self.lastWarnInst = -1
                return None

            # Warning Zone
            self.in_warning_zone = True
            if self.lastWarnInst == -1:
                self.lastWarnInst = self.instNum
                self.lastWarnPos = self.lastPos
        else:   # Neither DDM Drift nor Warning - disregard False warnings
            self.lastWarnInst = -1
            self.lastWarnPos  = -1

        if self.m_n > self.maxSizeConcept and (not self.in_warning_zone):  # RDDM Drift
            self.rddmDrift = True
