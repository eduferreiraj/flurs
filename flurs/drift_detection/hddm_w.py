import numpy as np
import sys
from .base_drift_detector import BaseDriftDetector

"""
 *    HDDM_W_Test.java
 *
 *    @author Isvani Frias-Blanco (ifriasb@udg.co.cu)
 *
 *    Licensed under the Apache License, Version 2.0 (the "License")
 *    you may not use self file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 *
 *

    Drift Detection Method based on the Hoeffding’s inequality.

    Parameters
    ----------
    _drift_confidence: float (default=0.001)
            Confidence to the drift

    _warning_confidence: float (default=0.005)
            Confidence to the warning

    _lambda: float (default=0.050)
            Controls how much weight is given to more recent data compared to older data. Smaller values mean less weight given to recent data.

    _oneside: bool (default=False)
            Monitors error increments and decrements (two-sided) or only increments (one-sided)

    Notes
    -----
    DDM (Drift Detection Method) [1]_ is a concept change detection method
    based on the PAC learning model premise, that the learner's error rate
    will decrease as the number of analysed samples increase, as long as the
    data distribution is stationary.

    If the algorithm detects an increase in the error rate, that surpasses
    a calculated threshold, either change is detected or the algorithm will
    warn the user that change may occur in the near future, which is called
    the warning zone.

    The detection threshold is calculated in function of two statistics,
    obtained when `(pi + si)` is minimum:

    * :math:`p_{min}`: The minimum recorded error rate.
    * `s_{min}`: The minimum recorded standard deviation.

    At instant :math:`i`, the detection algorithm uses:

    * :math:`p_i`: The error rate at instant i.
    * :math:`s_i`: The standard deviation at instant i.

    The conditions for entering the warning zone and detecting change are
    as follows:

    * if :math:`p_i + s_i \geq p_{min} + 2 * s_{min}` -> Warning zone
    * if :math:`p_i + s_i \geq p_{min} + 3 * s_{min}` -> Change detected

    References
    ----------
    .. [1] João Gama, Pedro Medas, Gladys Castillo, Pedro Pereira Rodrigues: Learning
       with Drift Detection. SBIA 2004: 286-295
 """

class HDDM_W(BaseDriftDetector):
    estimator_type = "drift_detector"

    def __init__(self, _drift_confidence = 0.001, _warning_confidence = 0.005, _lambda = 0.050, _oneside = True):
        super().__init__()

        self._drift_confidence = _drift_confidence
        self._warning_confidence  = _warning_confidence
        self._lambda = _lambda
        self._oneside = _oneside
        self.reset()

    def reset(self):
        super().reset()
        self.total = SampleInfo()
        self.sample1_DecrMonitoring = SampleInfo()
        self.sample1_IncrMonitoring = SampleInfo()
        self.sample2_DecrMonitoring = SampleInfo()
        self.sample2_IncrMonitoring = SampleInfo()
        self.incrCutPoint = sys.float_info.max
        self.decrCutPoint = sys.float_info.min
        self.width = 0
        self.delay = 0

    def add_element(self, value):
        auxDecayRate = 1.0 - self._lambda
        self.width += 1
        if self.total.EWMA_Estimator < 0:
            self.total.EWMA_Estimator = value
            self.total.independentBoundedConditionSum = 1
        else:
            self.total.EWMA_Estimator = self._lambda * value + auxDecayRate * self.total.EWMA_Estimator
            self.total.independentBoundedConditionSum = self._lambda * self._lambda + auxDecayRate * auxDecayRate * self.total.independentBoundedConditionSum

        self.updateIncrStatistics(value, self._drift_confidence)
        if self.monitorMeanIncr(value, self._drift_confidence):
            self.reset()
            self.in_concept_change = True
            self.in_warning_zone = False
        elif self.monitorMeanIncr(value, self._warning_confidence):
            self.in_concept_change = False
            self.in_warning_zone = True
        else:
            self.in_concept_change = False
            self.in_warning_zone = False

        self.updateDecrStatistics(value, self._drift_confidence)
        if not oneSidedTest and self.monitorMeanDecr(value, self._drift_confidence):
            self.reset()
        self.estimation = self.total.EWMA_Estimator

    def detectMeanIncrement(self, sample1, sample2, confidence):
        if sample1.EWMA_Estimator < 0 || sample2.EWMA_Estimator < 0:
            return False

        bound = np.sqrt((sample1.independentBoundedConditionSum + sample2.independentBoundedConditionSum) * np.log(1 / confidence) / 2)
        return sample2.EWMA_Estimator - sample1.EWMA_Estimator > bound

    def updateIncrStatistics(self, valor, confidence):
            auxDecay = 1.0 - self._lambda
            bound = np.sqrt(self.total.independentBoundedConditionSum * np.log(1.0 / self._drift_confidence) / 2)

            if self.total.EWMA_Estimator + bound < self.incrCutPoint:
                self.incrCutPoint = self.total.EWMA_Estimator + bound
                self.sample1_IncrMonitoring.EWMA_Estimator = self.total.EWMA_Estimator
                self.sample1_IncrMonitoring.independentBoundedConditionSum = self.total.independentBoundedConditionSum
                self.sample2_IncrMonitoring = SampleInfo()
                self.delay = 0
            else:
                self.delay += 1
                if self.sample2_IncrMonitoring.EWMA_Estimator < 0:
                    self.sample2_IncrMonitoring.EWMA_Estimator = valor
                    self.sample2_IncrMonitoring.independentBoundedConditionSum = 1
                else:
                    self.sample2_IncrMonitoring.EWMA_Estimator = self._lambda * valor + auxDecay * self.sample2_IncrMonitoring.EWMA_Estimator
                    self.sample2_IncrMonitoring.independentBoundedConditionSum = self._lambda * self._lambda + auxDecay * auxDecay * self.sample2_IncrMonitoring.independentBoundedConditionSum

    def monitorMeanIncr(self, valor, confidence):
        return detectMeanIncrement(self.sample1_IncrMonitoring, self.sample2_IncrMonitoring, confidence)

    def updateDecrStatistics(self, valor, confidence):
            auxDecay = 1.0 - self._lambda
            epsilon = np.sqrt(self.total.independentBoundedConditionSum * np.log(1.0 / self._drift_confidence) / 2)

            if self.total.EWMA_Estimator - epsilon > self.decrCutPoint:
                self.decrCutPoint = self.total.EWMA_Estimator - epsilon
                self.sample1_DecrMonitoring.EWMA_Estimator = self.total.EWMA_Estimator
                self.sample1_DecrMonitoring.independentBoundedConditionSum = self.total.independentBoundedConditionSum
                self.sample2_DecrMonitoring = SampleInfo()
            else:
                if self.sample2_DecrMonitoring.EWMA_Estimator < 0:
                    self.sample2_DecrMonitoring.EWMA_Estimator = valor
                    self.sample2_DecrMonitoring.independentBoundedConditionSum = 1
                else:
                    self.sample2_DecrMonitoring.EWMA_Estimator = self._lambda * valor + auxDecay * self.sample2_DecrMonitoring.EWMA_Estimator
                    self.sample2_DecrMonitoring.independentBoundedConditionSum = self._lambda * self._lambda + auxDecay * auxDecay * self.sample2_DecrMonitoring.independentBoundedConditionSum

    def monitorMeanDecr(valor, confidence):
        return detectMeanIncrement(self.sample2_DecrMonitoring, self.sample1_DecrMonitoring, confidence)

class SampleInfo():
    def __init__(self):
        self.EWMA_Estimator = -1.0
        self.independentBoundedConditionSum = 1
