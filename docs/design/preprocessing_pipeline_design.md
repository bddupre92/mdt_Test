# PreprocessingPipeline Design Document

## Overview

The `PreprocessingPipeline` class will provide a configurable, extensible pipeline for preprocessing data within the MoE framework. It will extend the existing preprocessing functionality in `data/preprocessing.py` while maintaining compatibility with the evolutionary computation aspects of the framework.

## Design Goals

1. **Configurability**: Allow users to define custom preprocessing steps and their order
2. **Extensibility**: Make it easy to add new preprocessing operations
3. **EC Compatibility**: Preserve feature characteristics needed by EC algorithms
4. **Reproducibility**: Ensure preprocessing steps can be serialized and reproduced
5. **Performance**: Optimize for efficient processing of large datasets
6. **Quality Awareness**: Provide quality metrics for each preprocessing step

## Class Structure

### `PreprocessingOperation` (Abstract Base Class)

```python
class PreprocessingOperation(ABC):
    """Abstract base class for all preprocessing operations."""
    
    @abstractmethod
    def fit(self, data: pd.DataFrame, **kwargs) -> None:
        """Fit the operation to the data."""
        pass
        
    @abstractmethod
    def transform(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Apply the operation to the data."""
        pass
        
    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        """Get the parameters of the operation."""
        pass
        
    @abstractmethod
    def set_params(self, params: Dict[str, Any]) -> None:
        """Set the parameters of the operation."""
        pass
        
    def fit_transform(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Fit and transform the data."""
        self.fit(data, **kwargs)
        return self.transform(data, **kwargs)
        
    def get_quality_metrics(self, data: pd.DataFrame, transformed_data: pd.DataFrame) -> Dict[str, float]:
        """Get quality metrics for the transformation."""
        return {}
```

### Concrete Operation Classes

#### `MissingValueHandler`

```python
class MissingValueHandler(PreprocessingOperation):
    """Handle missing values in the data."""
    
    def __init__(self, strategy: str = 'mean', fill_value: Optional[Any] = None, 
                 categorical_strategy: str = 'most_frequent', exclude_cols: List[str] = None):
        """Initialize the missing value handler."""
        self.strategy = strategy
        self.fill_value = fill_value
        self.categorical_strategy = categorical_strategy
        self.exclude_cols = exclude_cols or []
        self.fitted_imputers = {}
        
    def fit(self, data: pd.DataFrame, **kwargs) -> None:
        """Fit imputers to the data."""
        # Implementation details...
        
    def transform(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Impute missing values in the data."""
        # Implementation details...
        
    def get_params(self) -> Dict[str, Any]:
        """Get the parameters of the operation."""
        return {
            'strategy': self.strategy,
            'fill_value': self.fill_value,
            'categorical_strategy': self.categorical_strategy,
            'exclude_cols': self.exclude_cols
        }
        
    def set_params(self, params: Dict[str, Any]) -> None:
        """Set the parameters of the operation."""
        # Implementation details...
        
    def get_quality_metrics(self, data: pd.DataFrame, transformed_data: pd.DataFrame) -> Dict[str, float]:
        """Get quality metrics for the transformation."""
        # Implementation details...
```

#### `OutlierHandler`

```python
class OutlierHandler(PreprocessingOperation):
    """Detect and handle outliers in the data."""
    
    def __init__(self, method: str = 'zscore', threshold: float = 3.0, 
                 strategy: str = 'winsorize', exclude_cols: List[str] = None):
        """Initialize the outlier handler."""
        self.method = method
        self.threshold = threshold
        self.strategy = strategy
        self.exclude_cols = exclude_cols or []
        self.outlier_stats = {}
        
    def fit(self, data: pd.DataFrame, **kwargs) -> None:
        """Fit the outlier detection method to the data."""
        # Implementation details...
        
    def transform(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Detect and handle outliers in the data."""
        # Implementation details...
        
    def get_params(self) -> Dict[str, Any]:
        """Get the parameters of the operation."""
        return {
            'method': self.method,
            'threshold': self.threshold,
            'strategy': self.strategy,
            'exclude_cols': self.exclude_cols
        }
        
    def set_params(self, params: Dict[str, Any]) -> None:
        """Set the parameters of the operation."""
        # Implementation details...
        
    def get_quality_metrics(self, data: pd.DataFrame, transformed_data: pd.DataFrame) -> Dict[str, float]:
        """Get quality metrics for the transformation."""
        # Implementation details...
```

#### `FeatureScaler`

```python
class FeatureScaler(PreprocessingOperation):
    """Scale features in the data."""
    
    def __init__(self, method: str = 'minmax', feature_range: Tuple[float, float] = (0, 1), 
                 exclude_cols: List[str] = None):
        """Initialize the feature scaler."""
        self.method = method
        self.feature_range = feature_range
        self.exclude_cols = exclude_cols or []
        self.scalers = {}
        
    def fit(self, data: pd.DataFrame, **kwargs) -> None:
        """Fit scalers to the data."""
        # Implementation details...
        
    def transform(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Scale features in the data."""
        # Implementation details...
        
    def get_params(self) -> Dict[str, Any]:
        """Get the parameters of the operation."""
        return {
            'method': self.method,
            'feature_range': self.feature_range,
            'exclude_cols': self.exclude_cols
        }
        
    def set_params(self, params: Dict[str, Any]) -> None:
        """Set the parameters of the operation."""
        # Implementation details...
        
    def get_quality_metrics(self, data: pd.DataFrame, transformed_data: pd.DataFrame) -> Dict[str, float]:
        """Get quality metrics for the transformation."""
        # Implementation details...
```

#### `CategoryEncoder`

```python
class CategoryEncoder(PreprocessingOperation):
    """Encode categorical features in the data."""
    
    def __init__(self, method: str = 'label', exclude_cols: List[str] = None):
        """Initialize the category encoder."""
        self.method = method
        self.exclude_cols = exclude_cols or []
        self.encoders = {}
        
    def fit(self, data: pd.DataFrame, **kwargs) -> None:
        """Fit encoders to the data."""
        # Implementation details...
        
    def transform(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Encode categorical features in the data."""
        # Implementation details...
        
    def get_params(self) -> Dict[str, Any]:
        """Get the parameters of the operation."""
        return {
            'method': self.method,
            'exclude_cols': self.exclude_cols
        }
        
    def set_params(self, params: Dict[str, Any]) -> None:
        """Set the parameters of the operation."""
        # Implementation details...
        
    def get_quality_metrics(self, data: pd.DataFrame, transformed_data: pd.DataFrame) -> Dict[str, float]:
        """Get quality metrics for the transformation."""
        # Implementation details...
```

#### `FeatureSelector`

```python
class FeatureSelector(PreprocessingOperation):
    """Select features from the data."""
    
    def __init__(self, method: str = 'variance', threshold: float = 0.0, 
                 k: int = None, exclude_cols: List[str] = None):
        """Initialize the feature selector."""
        self.method = method
        self.threshold = threshold
        self.k = k
        self.exclude_cols = exclude_cols or []
        self.selected_features = []
        
    def fit(self, data: pd.DataFrame, **kwargs) -> None:
        """Fit the feature selection method to the data."""
        # Implementation details...
        
    def transform(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Select features from the data."""
        # Implementation details...
        
    def get_params(self) -> Dict[str, Any]:
        """Get the parameters of the operation."""
        return {
            'method': self.method,
            'threshold': self.threshold,
            'k': self.k,
            'exclude_cols': self.exclude_cols
        }
        
    def set_params(self, params: Dict[str, Any]) -> None:
        """Set the parameters of the operation."""
        # Implementation details...
        
    def get_quality_metrics(self, data: pd.DataFrame, transformed_data: pd.DataFrame) -> Dict[str, float]:
        """Get quality metrics for the transformation."""
        # Implementation details...
```

#### `TimeSeriesProcessor`

```python
class TimeSeriesProcessor(PreprocessingOperation):
    """Process time series data."""
    
    def __init__(self, time_col: str, resample_freq: str = None, 
                 lag_features: List[int] = None, exclude_cols: List[str] = None):
        """Initialize the time series processor."""
        self.time_col = time_col
        self.resample_freq = resample_freq
        self.lag_features = lag_features or []
        self.exclude_cols = exclude_cols or []
        
    def fit(self, data: pd.DataFrame, **kwargs) -> None:
        """Fit the time series processor to the data."""
        # Implementation details...
        
    def transform(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Process time series data."""
        # Implementation details...
        
    def get_params(self) -> Dict[str, Any]:
        """Get the parameters of the operation."""
        return {
            'time_col': self.time_col,
            'resample_freq': self.resample_freq,
            'lag_features': self.lag_features,
            'exclude_cols': self.exclude_cols
        }
        
    def set_params(self, params: Dict[str, Any]) -> None:
        """Set the parameters of the operation."""
        # Implementation details...
        
    def get_quality_metrics(self, data: pd.DataFrame, transformed_data: pd.DataFrame) -> Dict[str, float]:
        """Get quality metrics for the transformation."""
        # Implementation details...
```

### `PreprocessingPipeline` Class

```python
class PreprocessingPipeline:
    """A configurable pipeline for preprocessing data."""
    
    def __init__(self, operations: List[PreprocessingOperation] = None):
        """Initialize the preprocessing pipeline."""
        self.operations = operations or []
        self.quality_metrics = {}
        
    def add_operation(self, operation: PreprocessingOperation) -> None:
        """Add an operation to the pipeline."""
        self.operations.append(operation)
        
    def remove_operation(self, index: int) -> None:
        """Remove an operation from the pipeline."""
        if 0 <= index < len(self.operations):
            self.operations.pop(index)
            
    def fit(self, data: pd.DataFrame, **kwargs) -> None:
        """Fit the pipeline to the data."""
        for operation in self.operations:
            operation.fit(data, **kwargs)
            
    def transform(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Apply the pipeline to the data."""
        result = data.copy()
        self.quality_metrics = {}
        
        for i, operation in enumerate(self.operations):
            original = result.copy()
            result = operation.transform(result, **kwargs)
            
            # Collect quality metrics
            metrics = operation.get_quality_metrics(original, result)
            self.quality_metrics[f"step_{i}_{operation.__class__.__name__}"] = metrics
            
        return result
        
    def fit_transform(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Fit and transform the data."""
        self.fit(data, **kwargs)
        return self.transform(data, **kwargs)
        
    def get_params(self) -> Dict[str, Any]:
        """Get the parameters of the pipeline."""
        return {
            'operations': [
                {
                    'type': op.__class__.__name__,
                    'params': op.get_params()
                }
                for op in self.operations
            ]
        }
        
    def set_params(self, params: Dict[str, Any]) -> None:
        """Set the parameters of the pipeline."""
        if 'operations' in params:
            # Implementation details...
            pass
            
    def save(self, filepath: str) -> None:
        """Save the pipeline to a file."""
        # Implementation details...
        
    def load(self, filepath: str) -> None:
        """Load the pipeline from a file."""
        # Implementation details...
        
    def get_quality_summary(self) -> Dict[str, Any]:
        """Get a summary of quality metrics for the pipeline."""
        return self.quality_metrics
```

## Integration with Evolutionary Computation

The `PreprocessingPipeline` will integrate with the EC components of the MoE framework in the following ways:

1. **Feature Preservation**: The pipeline will preserve feature characteristics needed by EC algorithms, such as feature distributions and relationships.

2. **Quality Metrics**: The pipeline will provide quality metrics for each preprocessing step, which can be used by the Meta-Optimizer for algorithm selection.

3. **Feature Selection**: The `FeatureSelector` operation will integrate with the Ant Colony Optimization (ACO) algorithm for evolutionary feature selection.

4. **Parameter Optimization**: The pipeline parameters can be optimized using EC algorithms, such as Differential Evolution (DE) or Grey Wolf Optimizer (GWO).

## UI Integration

The `PreprocessingPipeline` will be integrated with the UI through a clean and functional interface built with Tailwind CSS. The UI will allow users to:

1. **Configure Pipeline**: Add, remove, and reorder preprocessing operations
2. **Set Parameters**: Configure parameters for each operation
3. **Preview Results**: See the effects of preprocessing on sample data
4. **Save/Load Pipelines**: Save and load pipeline configurations
5. **View Quality Metrics**: See quality metrics for each preprocessing step

## Implementation Plan

1. **Phase 1**: Implement the core `PreprocessingOperation` abstract class and the `PreprocessingPipeline` class
2. **Phase 2**: Implement the concrete operation classes
3. **Phase 3**: Integrate with the EC components
4. **Phase 4**: Create the UI for configuring the pipeline
5. **Phase 5**: Write comprehensive tests for all components

## Testing Strategy

1. **Unit Tests**: Test each operation class and the pipeline class in isolation
2. **Integration Tests**: Test the pipeline with different combinations of operations
3. **EC Integration Tests**: Test the integration with EC components
4. **UI Tests**: Test the UI for configuring the pipeline
5. **Performance Tests**: Test the pipeline with large datasets

## Conclusion

The `PreprocessingPipeline` class will provide a configurable, extensible pipeline for preprocessing data within the MoE framework. It will extend the existing preprocessing functionality while maintaining compatibility with the evolutionary computation aspects of the framework. The pipeline will be integrated with the UI through a clean and functional interface built with Tailwind CSS.
