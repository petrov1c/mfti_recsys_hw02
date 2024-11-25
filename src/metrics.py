from torchmetrics import MetricCollection, MeanSquaredError, MeanAbsoluteError


def get_metrics(**kwargs) -> MetricCollection:
    return MetricCollection(
        {
            'rmse': MeanSquaredError(squared=False, **kwargs),
            'mae': MeanAbsoluteError(**kwargs),
        }
    )
