WS_DIR := $(realpath ../..)

build:
	cd $(WS_DIR) && colcon build --symlink-install

clean:
	rm -rf $(WS_DIR)/build $(WS_DIR)/install $(WS_DIR)/log

format:
	black .
	
check_lint:
	flake8 --max-line-length=100 safe_bimanual_rl
	pylint --disable=E0401 safe_bimanual_rl