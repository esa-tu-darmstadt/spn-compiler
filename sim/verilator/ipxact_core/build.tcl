
ipx::infer_core -vendor esa.informatik.tu-darmstadt.de -name test.project -library user -taxonomy /UserIP -files ./src/spn_body.v -root_dir .
ipx::edit_ip_in_project -upgrade true -name edit_ip_project -directory tmp ./component.xml
ipx::current_core ./component.xml
set_property top spn_body [current_fileset]
set_property -quiet interface_mode monitor [ipx::get_bus_interfaces *MON* -of_objects [ipx::current_core]]
add_files ./src
update_compile_order -fileset sources_1
set_property name test.project [ipx::current_core]
set_property display_name test.project [ipx::current_core]
set_property description test.project [ipx::current_core]
set_property core_revision 1 [ipx::current_core]
set_property AUTO_FAMILY_SUPPORT_LEVEL level_1 [ipx::current_core]
foreach f {  } {
    set_property is_global_include true [get_files $f]
}
update_compile_order -fileset sources_1
update_compile_order -fileset sim_1
ipx::merge_project_changes files [ipx::current_core]
ipx::merge_project_changes ports [ipx::current_core]
puts "USED FILES"
foreach f [ipx::get_files -of_objects [ipx::get_file_groups *synthesis*]] {
    set n [get_property NAME $f]
    puts "USED FILE:$n"
}
puts "END USED FILES"
puts "Additional Parameters"

puts "End Additional Parameters"
ipx::create_xgui_files [ipx::current_core]
ipx::update_checksums [ipx::current_core]
ipx::save_core [ipx::current_core]
close_project -delete
puts "VIVADO FINISHED SUCCESSFULLY"
