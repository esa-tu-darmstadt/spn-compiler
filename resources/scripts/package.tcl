create_project -force $project_name ./$project_name
set project_dir [get_property directory [current_project]]
if {[string equal [get_filesets -quiet sources_1] ""]} {
  create_fileset -srcset sources_1
}

foreach {file} $files {
    add_files -fileset sources_1 $file
}
update_compile_order -fileset sources_1

create_bd_design $project_name
set M_AXI [create_bd_intf_port -mode Master -vlnv xilinx.com:interface:aximm_rtl:1.0 M_AXI]
set_property CONFIG.DATA_WIDTH $fulldata [get_bd_intf_ports $M_AXI]
set_property CONFIG.ADDR_WIDTH $fulladdr [get_bd_intf_ports $M_AXI]
set S_AXI_LITE [create_bd_intf_port -mode Slave -vlnv xilinx.com:interface:aximm_rtl:1.0 S_AXI_LITE]
set_property CONFIG.DATA_WIDTH $litedata [get_bd_intf_ports $S_AXI_LITE]
set_property CONFIG.ADDR_WIDTH $liteaddr [get_bd_intf_ports $S_AXI_LITE]
set_property CONFIG.PROTOCOL AXI4LITE [get_bd_intf_ports $S_AXI_LITE]
#set AXI4_STREAM_MASTER [create_bd_intf_port -mode Master -vlnv xilinx.com:interface:axis_rtl:1.0 AXI4_STREAM_MASTER]
#set AXI4_STREAM_SLAVE  [create_bd_intf_port -mode Slave -vlnv xilinx.com:interface:axis_rtl:1.0 AXI4_STREAM_SLAVE]
#set_property CONFIG.TDATA_NUM_BYTES $streamin_bytes [get_bd_intf_ports $AXI4_STREAM_SLAVE]
#set_property CONFIG.TDATA_NUM_BYTES $streamout_bytes [get_bd_intf_ports $AXI4_STREAM_MASTER]

proc create_inverter {name} {
      variable ret [create_bd_cell -type ip -vlnv xilinx.com:ip:util_vector_logic:2.0 $name]
      set_property -dict [list CONFIG.C_SIZE {1} CONFIG.C_OPERATION {not} CONFIG.LOGO_FILE {data/sym_notgate.png}] [get_bd_cells $name]
      return $ret
    }

set ap_clk   [ create_bd_port -dir I -type clk ap_clk ]
set ap_rst_n [ create_bd_port -dir I -type rst ap_rst_n ]
set out_inv  [ create_inverter reset_inverter]
set interrupt [ create_bd_port -dir O -type intr interrupt ]

set_property CONFIG.POLARITY ACTIVE_LOW $ap_rst_n

set TAP [create_bd_cell -type module -reference AXI4StreamMapper TAP]
set_property CONFIG.POLARITY ACTIVE_HIGH [get_bd_pins $TAP/reset]

#set DWCS [create_bd_cell -type ip -vlnv xilinx.com:ip:axis_dwidth_converter:1.1 DWCS]
#set_property -dict [list CONFIG.S_TDATA_NUM_BYTES.VALUE_SRC USER] [get_bd_cells $DWCS]
#set_property CONFIG.S_TDATA_NUM_BYTES $streamin_bytes [get_bd_cells $DWCS]
#set_property CONFIG.M_TDATA_NUM_BYTES $fulldata_bytes [get_bd_cells $DWCS]

#set DWCM [create_bd_cell -type ip -vlnv xilinx.com:ip:axis_dwidth_converter:1.1 DWCM]
#set_property -dict [list CONFIG.S_TDATA_NUM_BYTES.VALUE_SRC USER] [get_bd_cells $DWCM]
#set_property CONFIG.S_TDATA_NUM_BYTES $fulldata_bytes [get_bd_cells $DWCM]
#set_property CONFIG.M_TDATA_NUM_BYTES $streamout_bytes [get_bd_cells $DWCM]

connect_bd_net [get_bd_port $ap_rst_n] [get_bd_pins $out_inv/Op1]
connect_bd_net [get_bd_pins $out_inv/Res] [get_bd_pins $TAP/reset]
connect_bd_net [get_bd_ports $interrupt] [get_bd_pins $TAP/interrupt]
connect_bd_net [get_bd_ports $ap_clk] [get_bd_pins $TAP/clock]
#connect_bd_net [get_bd_port $ap_rst_n] [get_bd_pins $DWCM/aresetn]
#connect_bd_net [get_bd_port $ap_rst_n] [get_bd_pins $DWCS/aresetn]
#connect_bd_net [get_bd_port $ap_clk] [get_bd_pins $DWCM/aclk]
#connect_bd_net [get_bd_port $ap_clk] [get_bd_pins $DWCS/aclk]

connect_bd_intf_net [get_bd_intf_ports $S_AXI_LITE] [get_bd_intf_pins $TAP/S_AXI_LITE]
connect_bd_intf_net [get_bd_intf_ports $M_AXI] [get_bd_intf_pins $TAP/M_AXI]
#connect_bd_intf_net [get_bd_intf_ports $AXI4_STREAM_SLAVE] [get_bd_intf_pins $DWCS/S_AXIS]
#connect_bd_intf_net [get_bd_intf_pins $DWCS/M_AXIS] [get_bd_intf_pins $TAP/S_AXIS]
#connect_bd_intf_net [get_bd_intf_pins $TAP/M_AXIS] [get_bd_intf_pins $DWCM/S_AXIS]
#connect_bd_intf_net [get_bd_intf_ports $AXI4_STREAM_MASTER] [get_bd_intf_pins $DWCM/M_AXIS]

assign_bd_address
set_property range 64K [get_bd_addr_segs {S_AXI_LITE/SEG_TAP_reg0}]
set_property offset 0x00000000 [get_bd_addr_segs {TAP/M_AXI/SEG_M_AXI_Reg}]
set_property range 4G [get_bd_addr_segs {TAP/M_AXI/SEG_M_AXI_Reg}]
validate_bd_design

set bd_file [get_files ${project_name}.bd]
make_wrapper -files $bd_file -top
add_files -norecurse [file join [file dirname $bd_file] hdl/${project_name}_wrapper.v]
set_property synth_checkpoint_mode Singular $bd_file
generate_target all $bd_file

ipx::package_project -root_dir . -module ${project_name} -generated_files -import_files -force
set_property supported_families {qvirtex7 Beta virtexuplus Beta virtexuplusHBM Beta qzynq Beta zynquplus Beta kintexu Beta versal Beta virtex7 Beta virtexu Beta azynq Beta zynq Beta} [ipx::current_core]
set core [ipx::current_core]
set sim_fg [ipx::get_file_groups xilinx_anylanguagebehavioralsimulation]
foreach file [ipx::get_files -type systemCSource -of_objects $sim_fg] {
      ipx::remove_file [get_property NAME $file] -file_group $sim_fg
}

set_property vendor $vendor $core
set_property library $library $core
set_property display_name "$project_name Free Running Kernel" $core
set_property vendor_display_name $vendor $core

set_property name ap_clk [ipx::get_bus_interfaces CLK.AP_CLK -of_objects $core]
ipx::remove_bus_parameter FREQ_HZ [ipx::get_bus_interfaces ap_clk -of_objects $core]
ipx::remove_bus_parameter PHASE [ipx::get_bus_interfaces ap_clk -of_objects $core]
set_property name ap_rst_n [ipx::get_bus_interfaces RST.AP_RST_N -of_objects $core]
set_property name interrupt [ipx::get_bus_interfaces INTR.INTERRUPT -of_objects $core]

set_property sdx_kernel true $core
set_property sdx_kernel_type rtl $core

ipx::create_xgui_files $core
ipx::update_checksums $core
ipx::save_core $core
ipx::check_integrity -quiet $core
ipx::archive_core ./${project_name}_1.0.zip $core

ipx::unload_core component_1
exit
