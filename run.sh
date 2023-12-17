#!/bin/bash

python ./fate_flow/fate_flow_client.py -f upload -c ./examples/federatedml-1.x-examples/hetero_nn/upload_data_guest.json
python ./fate_flow/fate_flow_client.py -f upload -c ./examples/federatedml-1.x-examples/hetero_nn/upload_data_host.json
python ./fate_flow/fate_flow_client.py -f submit_job -c ./examples/federatedml-1.x-examples/hetero_nn/test_hetero_nn_keras_multi_label.json -d ./examples/federatedml-1.x-examples/hetero_nn/test_hetero_nn_dsl.json