WS_DIR := $(realpath ../..)

build:
	cd $(WS_DIR) && colcon build --symlink-install

clean:
	rm -rf $(WS_DIR)/build $(WS_DIR)/install $(WS_DIR)/log

format:
	black .
	
check_lint:
	flake8 --max-line-length=100 safe_bimanual_rl
	pylint --disable=E0401,C0114,C0103 safe_bimanual_rl

visualise:
	python3 safe_bimanual_rl/environments/visualise.py

visualise_controller:
	python3 safe_bimanual_rl/utils/sinusoidal_controller.py