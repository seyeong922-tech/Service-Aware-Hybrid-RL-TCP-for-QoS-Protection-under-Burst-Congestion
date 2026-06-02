function(build_qos_rl_tcp_exec target_name source_file)
  build_exec(
    EXECNAME ${target_name}
    SOURCE_FILES
      ${source_file}
      gym_tcp_env.cc
    LIBRARIES_TO_LINK
      "${ns3-libs}"
      "${ns3-contrib-libs}"
    EXECUTABLE_DIRECTORY_PATH
      ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/scratch/qos_rl_tcp/
  )
endfunction()

build_qos_rl_tcp_exec(topo3_parkinglot topo3_parkinglot.cc)
build_qos_rl_tcp_exec(topo3_dumbbell topo3_dumbbell.cc)
build_qos_rl_tcp_exec(topo3_dumbbell_asym topo3_dumbbell_asym.cc)
build_qos_rl_tcp_exec(topo3_star_gateway topo3_star_gateway.cc)
build_qos_rl_tcp_exec(topo3_parkinglot_stress topo3_parkinglot_stress.cc)
