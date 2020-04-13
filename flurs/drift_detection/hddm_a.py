"""
 *    HDDM_A_Test.java
 *
 *    @author Isvani Frias-Blanco (ifriasb@udg.co.cu)
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
 *
 *
 """


import numpy as np
from .base_drift_detector import BaseDriftDetector

"""*
 * <p>Online drift detection method based on Hoeffding's bounds.
 * HDDM<sub><i>A</i>-test</sub> uses the average as estimator.
 * It receives as input a stream of real values and returns the estimated status
 * of the stream: STABLE, WARNING or DRIFT.</p>
 *
 * <p>I. Frias-Blanco, J. del Campo-Avila, G. Ramos-Jimenez, R. Morales-Bueno,
 * A. Ortiz-Diaz, and Y. Caballero-Mota, Online and non-parametric drift
 * detection methods based on Hoeffding's bound, IEEE Transactions on Knowledge
 * and Data Engineering, 2014. DOI 10.1109/TKDE.2014.2345382.</p>
 *
 * <p>Parameters:</p> <ul> <li>-d : Confidence to the drift</li><li>-w :
 * Confidence to the warning</li><li>-t : Option to monitor error increments and
 * decrements (two-sided) or only increments (one-sided)</li>
 * </ul>
 *
 * @author Isvani Frias-Blanco (ifriasb@udg.co.cu)
 *
 """

class HDDM_A_Test(AbstractChangeDetector):

    # public FloatOption driftConfidenceOption = new FloatOption("driftConfidence", 'd',
    #         "Confidence to the drift",
    #         0.001, 0, 1)
    # public FloatOption warningConfidenceOption = new FloatOption("warningConfidence", 'w',
    #         "Confidence to the warning",
    #         0.005, 0, 1)
    # public MultiChoiceOption oneSidedTestOption = new MultiChoiceOption(
    #         "typeOfTest", 't',
    #         "Monitors error increments and decrements (two-sided) or only increments (one-sided)", new String[]{
    #     "One-sided", "Two-sided"}, new String[]{
    #     "One-sided", "Two-sided"},
    #         1)

    def __init__(self, _drift_confidence = 0.001, _warning_confidence = 0.005, _lambda = 0.050, _oneside = True):
        super().__init__()

        self._drift_confidence = _drift_confidence
        self._warning_confidence  = _warning_confidence
        self._lambda = _lambda
        self._oneside = _oneside
        self.reset()

    def reset(self):
        super.resetLearning()
        self.n_min = 0
        self.c_min = 0
        self.total_n = 0
        self.total_c = 0
        self.n_max = 0
        self.c_max = 0
        self.cEstimacion = 0
        self.nEstimacion = 0


    def add_element(self, value):
        self.total_n++
        self.total_c += value
        if self.n_min == 0:
            self.n_min = self.total_n
            self.c_min = self.total_c

        if self.n_max == 0:
            self.n_max = self.total_n
            self.c_max = self.total_c

        cota = np.sqrt(1.0 / (2 * self.n_min) * np.log(1.0 / self._drift_confidence))
        cota1 = np.sqrt(1.0 / (2 * self.total_n) * np.log(1.0 / self._drift_confidence))

        if self.c_min / self.n_min + cota >= self.total_c / self.total_n + cota1:
            self.c_min = self.total_c
            self.n_min = self.total_n

        cota = np.sqrt(1.0 / (2 * self.n_max) * np.log(1.0 / self._drift_confidence))
        if self.c_max / self.n_max - cota <= self.total_c / self.total_n - cota1:
            self.c_max = self.total_c
            self.n_max = self.total_n

        if self.meanIncr(self.c_min, self.n_min, self.total_c, self.total_n, self._drift_confidence):
            self.nEstimacion = self.total_n - self.n_min
            self.cEstimacion = self.total_c - self.c_min
            self.n_min = self.n_max = self.total_n = 0
            self.c_min = self.c_max = self.total_c = 0
            self.in_concept_change = True
            self.in_warning_zone = False

        elif self.meanIncr(self.c_min, self.n_min, self.total_c, self.total_n, self._warning_confidence):
            self.in_concept_change = False
            self.in_warning_zone = True
        else:
            self.in_concept_change = False
            self.in_warning_zone = False

        if self._oneside and self.meanDecr(self.c_max, self.n_max, self.total_c, self.total_n):
            self.nEstimacion = self.total_n - self.n_max
            self.cEstimacion = self.total_c - self.c_max
            self.n_min = 0
            self.n_max = 0
            self.total_n = 0
            self.c_min = 0
            self.c_max = 0
            self.total_c = 0
        self.updateEstimations()

    def meanIncr(self, c_min, n_min, total_c, total_n, confianzaCambio):
        if n_min == total_n:
            return False

        m = float((total_n - n_min) / n_min * (1.0 / total_n))
        cota = np.sqrt(m / 2 * np.log(2.0 / confianzaCambio))
        return total_c / total_n - c_min / n_min >= cota

    def meanDecr(self, c_max, n_max, total_c, total_n):
        if n_max == total_n:
            return False

        m = float((total_n - n_max) / n_max * (1.0 / total_n))
        cota = np.sqrt(m / 2 * np.log(2.0 / self._drift_confidence))
        return c_max / n_max - total_c / total_n >= cota

    def updateEstimations(self):
        if self.total_n >= self.nEstimacion:
            self.cEstimacion = 0
            self.nEstimacion = 0
            self.estimation = self.total_c / self.total_n
            self.delay = self.total_n
        else:
            self.estimation = self.cEstimacion / self.nEstimacion
            self.delay = self.nEstimacion
