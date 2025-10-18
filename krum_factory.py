import tensorflow as tf
import tensorflow_federated as tff


class KrumFactory(tff.aggregators.UnweightedAggregationFactory):
    """Aggregator for the KRUM algorithm."""
    def create(self, value_type):

        @tff.federated_computation()
        def init_fn():
            return tff.federated_value((), tff.SERVER)

        @tff.federated_computation(init_fn.type_signature.result,
            tff.FederatedType(value_type, tff.CLIENTS),)
        def next_fn(state, value):
            
            return tff.templates.MeasuredProcessOutput(state, None, None)

        return tff.templates.AggregationProcess(init_fn, next_fn)
    
