function(lotus_protobuf_generate)
  include(CMakeParseArguments)
  if(EXISTS "${ONNX_CUSTOM_PROTOC_EXECUTABLE}")
    set(LOTUS_PROTOC_EXECUTABLE ${ONNX_CUSTOM_PROTOC_EXECUTABLE})
  else()
    set(LOTUS_PROTOC_EXECUTABLE $<TARGET_FILE:protobuf::protoc>)
    set(LOTUS_PROTOC_DEPS protobuf::protoc)
  endif()
  set(_options APPEND_PATH)
  set(_singleargs LANGUAGE OUT_VAR EXPORT_MACRO)
  if(COMMAND target_sources)
    list(APPEND _singleargs TARGET)
  endif()
  set(_multiargs PROTOS IMPORT_DIRS GENERATE_EXTENSIONS)

  cmake_parse_arguments(lotus_protobuf_generate "${_options}" "${_singleargs}" "${_multiargs}" "${ARGN}")

  if(NOT lotus_protobuf_generate_PROTOS AND NOT lotus_protobuf_generate_TARGET)
    message(SEND_ERROR "Error: lotus_protobuf_generate called without any targets or source files")
    return()
  endif()

  if(NOT lotus_protobuf_generate_OUT_VAR AND NOT lotus_protobuf_generate_TARGET)
    message(SEND_ERROR "Error: lotus_protobuf_generate called without a target or output variable")
    return()
  endif()

  if(NOT lotus_protobuf_generate_LANGUAGE)
    set(lotus_protobuf_generate_LANGUAGE cpp)
  endif()
  string(TOLOWER ${lotus_protobuf_generate_LANGUAGE} lotus_protobuf_generate_LANGUAGE)

  if(lotus_protobuf_generate_EXPORT_MACRO AND lotus_protobuf_generate_LANGUAGE STREQUAL cpp)
    set(_dll_export_decl "dllexport_decl=${lotus_protobuf_generate_EXPORT_MACRO}:")
  endif()

  if(NOT lotus_protobuf_generate_EXTENSIONS)
    if(lotus_protobuf_generate_LANGUAGE STREQUAL cpp)
      set(lotus_protobuf_generate_EXTENSIONS .pb.h .pb.cc)
    elseif(lotus_protobuf_generate_LANGUAGE STREQUAL python)
      set(lotus_protobuf_generate_EXTENSIONS _pb2.py)
    else()
      message(SEND_ERROR "Error: lotus_protobuf_generate given unknown Language ${LANGUAGE}, please provide a value for GENERATE_EXTENSIONS")
      return()
    endif()
  endif()

  if(lotus_protobuf_generate_TARGET)
    get_target_property(_source_list ${lotus_protobuf_generate_TARGET} SOURCES)
    foreach(_file ${_source_list})
      if(_file MATCHES "proto$")
        list(APPEND lotus_protobuf_generate_PROTOS ${_file})
      endif()
    endforeach()
  endif()

  if(NOT lotus_protobuf_generate_PROTOS)
    message(SEND_ERROR "Error: lotus_protobuf_generate could not find any .proto files")
    return()
  endif()

  if(lotus_protobuf_generate_APPEND_PATH)
    # Create an include path for each file specified
    foreach(_file ${lotus_protobuf_generate_PROTOS})
      get_filename_component(_abs_file ${_file} ABSOLUTE)
      get_filename_component(_abs_path ${_abs_file} PATH)
      list(FIND _protobuf_include_path ${_abs_path} _contains_already)
      if(${_contains_already} EQUAL -1)
          list(APPEND _protobuf_include_path -I ${_abs_path})
      endif()
    endforeach()
  else()
    set(_protobuf_include_path -I ${CMAKE_CURRENT_SOURCE_DIR})
  endif()

  foreach(DIR ${lotus_protobuf_generate_IMPORT_DIRS})
    get_filename_component(ABS_PATH ${DIR} ABSOLUTE)
    list(FIND _protobuf_include_path ${ABS_PATH} _contains_already)
    if(${_contains_already} EQUAL -1)
        list(APPEND _protobuf_include_path -I ${ABS_PATH})
    endif()
  endforeach()

  set(_generated_srcs_all)
  foreach(_proto ${lotus_protobuf_generate_PROTOS})
    get_filename_component(_abs_file ${_proto} ABSOLUTE)
    get_filename_component(_basename ${_proto} NAME_WE)

    set(_generated_srcs)
    foreach(_ext ${lotus_protobuf_generate_EXTENSIONS})
      list(APPEND _generated_srcs "${CMAKE_CURRENT_BINARY_DIR}/${_basename}${_ext}")
    endforeach()
    list(APPEND _generated_srcs_all ${_generated_srcs})

    add_custom_command(
      OUTPUT ${_generated_srcs}
      COMMAND  ${LOTUS_PROTOC_EXECUTABLE}
      ARGS --${lotus_protobuf_generate_LANGUAGE}_out ${_dll_export_decl}${CMAKE_CURRENT_BINARY_DIR} ${_protobuf_include_path} ${_abs_file}
      DEPENDS ${_abs_file} ${LOTUS_PROTOC_DEPS}
      COMMENT "Running ${lotus_protobuf_generate_LANGUAGE} protocol buffer compiler on ${_proto}"
      VERBATIM )
  endforeach()

  set_source_files_properties(${_generated_srcs_all} PROPERTIES GENERATED TRUE)
  if(lotus_protobuf_generate_OUT_VAR)
    set(${lotus_protobuf_generate_OUT_VAR} ${_generated_srcs_all} PARENT_SCOPE)
  endif()
  if(lotus_protobuf_generate_TARGET)
    target_sources(${lotus_protobuf_generate_TARGET} PRIVATE ${_generated_srcs_all})
  endif()

endfunction()
