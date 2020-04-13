"""
 *    DDM.java
 *    Copyright (C) 2008 University of Waikato, Hamilton, New Zealand
 *    @author Manuel Baena (mbaena@lcc.uma.es)
 *
 *    This program is free software you can redistribute it and/or modify
 *    it under the terms of the GNU General License as published by
 *    the Free Software Foundation either version 3 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General License for more details.
 *
 *    You should have received a copy of the GNU General License
 *    along with this program. If not, see <http:#www.gnu.org/licenses/>.
 """
import numpy as np
from .base_drift_detector import BaseDriftDetector

"""*
 * Drift detection method based in Geometric Moving Average Test
 *
 *
 * @author Manuel Baena (mbaena@lcc.uma.es)
 * @version $Revision: 7 $
 """
class GeometricMovingAverageDM(AbstractChangeDetector {

    private static final long serialVersionUID = -3518369648142099719L

    IntOption minNumInstancesOption = new IntOption(
            "minNumInstances",
            'n',
            "The minimum number of instances before permitting detecting change.",
            30, 0, Integer.MAX_VALUE)

    FloatOption lambdaOption = new FloatOption("lambda", 'l',
            "Threshold parameter of the Geometric Moving Average Test", 1, 0.0, Float.MAX_VALUE)

    FloatOption alphaOption = new FloatOption("alpha", 'a',
            "Alpha parameter of the Geometric Moving Average Test", .99, 0.0, 1.0)

    private m_n

    private sum

    private x_mean

    private alpha

    private delta

    private lambda

    GeometricMovingAverageDM(:
        resetLearning()
    }

    @Override
    def reset(self:
        m_n = 1.0
        x_mean = 0.0
        sum = 0.0
        alpha = self.alphaOption.getValue()
        lambda = self.lambdaOption.getValue()
    }

    @Override
    void input(x:
        # It monitors the error rate
        if self.in_concept_change == True || self.isInitialized == False:
            resetLearning()
            self.isInitialized = True
        }

        x_mean = x_mean + (x - x_mean) / m_n
        sum = alpha * sum + ( 1.0- alpha) * (x - x_mean)


        m_n += 1

        # System.out.print(prediction + " " + m_n + " " + (m_p+m_s) + " ")
        self.estimation = x_mean
        self.in_concept_change = False
        self.in_warning_zone = False
        self.delay = 0

        if m_n < self.minNumInstancesOption.getValue():
            return
        }

        if sum > self.lambda:
            self.in_concept_change = True
        }
    }

    @Override
    void getDescription(StringBuilder sb, int indent:
        # TODO Auto-generated method stub
    }

    @Override
    protected void prepareForUseImpl(TaskMonitor monitor,
            ObjectRepository repository:
        # TODO Auto-generated method stub
    }
}
