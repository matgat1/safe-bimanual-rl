WS_DIR := $(realpath ../..)

build:
	cd $(WS_DIR) && colcon build --symlink-install

clean:
	rm -rf $(WS_DIR)/build $(WS_DIR)/install $(WS_DIR)/log
