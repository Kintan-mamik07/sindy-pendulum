import numpy as np
from pysindy import SINDy
from pysindy.feature_library import (
    PolynomialLibrary,
    FourierLibrary,
    GeneralizedLibrary,
    CustomLibrary,
)
from pysindy.optimizers import STLSQ, SR3
from typing import Dict, Any, List

class SINDyWrapper:
        def __init__(
            self,
            feature_library: str = "poly",
            optimizer: str = "stlsq",
            threshold: float = 0.05,
            degree: int = 3,
            t=None,
            **kwargs,
        ):
            self.feature_library = feature_library
            self.optimizer = optimizer
            self.threshold = threshold
            self.degree = degree
            self.kwargs = kwargs
            self.model=None # Initialize the model attribute
            self.metrics= {} # Initialize the metrics attribute

            lib=self._build_library(feature_library, degree,t) #Build the feature library based on the specified type and degree
            opt=self._build_optimizer(optimizer, threshold) #Build the optimizer based on the specified type and threshold

            #create the SINDy model with the specified library and optimizer, and any additional keyword arguments

            from pysindy.differentiation import SmoothedFiniteDifference
            self.model = SINDy(
            feature_library=lib,
            optimizer=opt,
            differentiation_method=SmoothedFiniteDifference(),
    **kwargs
)


        def _build_library(self, lib_type: str, degree: int,t=None):
            import pysindy as ps
            import numpy as np
            if lib_type in ["poly", "polynomial"]:
                return ps.PolynomialLibrary(degree=degree)
            elif lib_type == "fourier":
                return ps.FourierLibrary(n_frequencies=3)
            elif lib_type == "combined":
                return ps.GeneralizedLibrary(
                    libraries=[ps.PolynomialLibrary(degree=degree), ps.FourierLibrary(n_frequencies=3)]
                )
            
            elif lib_type == "custom":
                functions = [
        lambda x: np.sin(x),
        lambda x: np.cos(x),
        lambda x, y: np.sin(x - y),
        lambda x, y: np.cos(x - y),
        lambda x, y: np.sin(x - y) * x,
        lambda x, y: np.cos(x - y) * y**2,
    ]
                function_names = [
        lambda x: f"sin({x})",
        lambda x: f"cos({x})",
        lambda x, y: f"sin({x}-{y})",
        lambda x, y: f"cos({x}-{y})",
        lambda x, y: f"sin({x}-{y})*{x}",
        lambda x, y: f"cos({x}-{y})*{y}^2",
    ]
                return ps.feature_library.CustomLibrary(
        library_functions=functions,
        function_names=function_names,
    )
            
            elif lib_type =="implicit":
                if t is None:
                    raise ValueError("Time array 't' must be provided for implicit library.")
                functions =[
                    lambda x:x,
                    lambda x,y:x*y,
                    lambda x: x**2,
                    lambda x: np.sin(x),
                    lambda x: np.cos(x)
                ]
                function_names = [
                    lambda x: x,
                    lambda x,y:x+"*" +y,
                    lambda x: x+"^2",
                    lambda x: "sin("+x+")",
                    lambda x: "cos("+x+")"
                ]

                base_lib=ps.feature_library.CustomLibrary(
                    library_functions=functions,
                    function_names=function_names,
                )
                return ps.PDELibrary(
                    function_library=base_lib,
                    derivative_order=0,
                    implicit_terms=True,
                    include_bias=True,
                    temporal_grid=t
                )
            else:
                raise ValueError(f"Unknown library type: {lib_type}")
            


        def _build_optimizer(self, opt_type: str, threshold: float):
            if opt_type == "stlsq":
                return STLSQ(threshold=threshold, alpha=1e-5)
            elif opt_type == "sr3":
                return SR3(threshold=threshold)
            elif opt_type == "sindypi":
                from pysindy.optimizers import SINDyPI
                return SINDyPI(
                    reg_weight_lam=threshold,
                    regularizer="l1",
                    tol=1e-5,
                    max_iter=10000
                )
            else:
                raise ValueError(f"Unknown optimizer type: {opt_type}")

        def fit(
                self,
                X: list, # Changed to list since we are passing a list of trajectories
                t: np.ndarray = None,
                dt: np.ndarray = None,
                quiet: bool = False,
                **kwargs # Catch any extra arguments like multiple_trajectories just in case
        ) -> "SINDyWrapper":
            self.model.fit(X, t=t)  #pysindy doesnt support dt parameter in fit method, so we pass t instead of dt and quiet parameter is not used in the fit method of pysindy, so we ignore it
            return self
        

        def predict(self, x:np.ndarray, t:np.ndarray) -> np.ndarray:
            return self.model.predict(x, t=t)
        
        #def score(self,x:np.ndarray, t:np.ndarray=None,dt: float= None) -> float:
          #  return self.model.score(x, t=t)  #R² score of the model on the provided data. It measures how well the model's predictions match the actual data, with a score of 1 indicating perfect prediction and lower scores indicating worse performance.
        
        def score(self, x, t=None):
           import numpy as np
           from pysindy.optimizers import SINDyPI
        
        # SINDy-PI outputs implicit equations for every library term (23 terms),
        # not just the 4 state derivatives. Standard R^2 cannot be computed.
           if isinstance(self.model.optimizer, SINDyPI):
              print("\n[Metrics] Skipping R² score: Standard R² is not applicable to SINDy-PI implicit models.")
              return np.nan
            
           return self.model.score(x, t=t)
        
        def get_equations(self) -> List[str]:

            try:
                return self.model.equations()
            except TypeError:
                return self.model.equations(precision=4)  # Return equations with specified precision if TypeError occurs
            
        def print_equations(self, precision: int = 4):
            try:
                equations = self.model.equations()
                print(f"Equations with default precision:\n{equations}")
            except:
                if hasattr(self.model, 'equations'):
                    equations = self.model.equations(precision=precision)
                    print(f"Equations with precision {precision}:\n{equations}")   

        def get_coefficients(self) ->np.ndarray:
            return self.model.coefficients()

        def get_feauture_names(self) -> List[str]:
            return self.model.get_feature_names()

        def compute_metrics(self, X_test: np.ndarray, t_test: np.ndarray) ->Dict[str,float]:
            train_score = self.score(X_test,t=t_test)

            # Count nonzero coefficients
            coeffs = self.get_coefficients()
            n_active = np.count_nonzero(coeffs)
            sparsity = 1.0 - (n_active / coeffs.size)
        
            self.metrics = {
            "r2_score": train_score,
            "n_active_features": int(n_active),
            "sparsity": float(sparsity),
            "total_features": coeffs.size,
            }
        
            return self.metrics      

        def summary(self) -> Dict[str, Any]:
          """Get model summary."""
          return {
            "library": self.feature_library,
            "optimizer": self.optimizer,
            "threshold": self.threshold,
            "degree": self.degree,
            "metrics": self.metrics,
            "equations": self.get_equations(),
        }  
        @staticmethod 

        def compare_libraries(
                X: np.ndarray,
                t: np.ndarray,
                libraries: List[str] = None,
                thresholds: List[float] = None,
        ) ->Dict[str,Dict]:
            
            if libraries is None:
                libraries = ["poly", "fourier", "combined"]
            if thresholds is None:
                thresholds = [0.01, 0.05, 0.1]
    
            results = {}

            for lib in libraries:
                results[lib] = {}
                for thresh in thresholds:
                    for thresh in thresholds:
                        model = SINDyWrapper(feature_library=lib, threshold=thresh,degree=3)
                        model.fit(X, t=t,quiet=True)
                        metrics = model.compute_metrics(X, t)
                        results[lib][thresh] = metrics

            return results
