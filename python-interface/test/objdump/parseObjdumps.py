# ==============================================================================
#  This file is part of the SPNC project under the Apache License v2.0 by the
#  Embedded Systems and Applications Group, TU Darmstadt.
#  For the full copyright and license information, please view the LICENSE
#  file that was distributed with this source code.
#  SPDX-License-Identifier: Apache-2.0
# ==============================================================================

import csv
import fire
import os
import re
import subprocess


def parse_output(output, vectorized):
    arithmetic_insts = {
        "add",
        "sub",
        "vaddss",
        "vaddsd",
        "vaddps",
        "vaddpd",
        "vpaddq",
        "vsubss",
        "vsubsd",
        "vsubps",
        "vsubpd",
        "vmulss",
        "vmulsd",
        "vmulps",
        "vmulpd",
        "vdivpd",
        "vminss",
        "vminps",
        "vmaxss",
        "vmaxps",
        "vzeroupper",
    }
    move_insts = {
        "mov",
        "vmovss",
        "vmovd",
        "vmovq",
        "vmovupd",
        "vmovddup",
        "vmovdqu",
        "vmovsldup",
        "vmovlps",
        "vmovlhps",
        "vmovups",
        "vmovapd",
        "vmovaps",
        "vmovdqa",
        "vmovshdup",
        "vpmovzxdq",
    }
    vector_packing_insts = {
        "vbroadcastss",
        "vbroadcastsd",
        "vpbroadcastd",
        "vpbroadcastq",
        "vbroadcasti128",
        "vbroadcastf128",
        "vpextrq",
        "vgatherdpd",
        "vinserti128",
        "vinsertf128",
        "vinsertps",
        "vextracti128",
        "vextractf128",
        "vunpcklps",
        "vunpcklpd",
        "vunpckhps",
        "vunpckhpd",
        "vpunpckldq",
        "vpunpckhdq",
        "vpunpckhqdq",
        "vpermps",
        "vpermpd",
        "vpermd",
        "vpermq",
        "vperm2i128",
        "vperm2f128",
        "vshufps",
        "vshufpd",
        "vpshufd",
        "vpermilpd",
        "vpermilps",
        "vxorps",  # used to fill a vector with zeros: vxorps xmm0, xmm0, xmm0
        "vpcmpeqd",  # used to fill a vector with ones: vpcmpeqd ymm0, ymm0, ymm0
        "vpblendd",
        "vblendps",
        "vblendpd",
        "vpalignr",
    }
    other_insts = {
        "push",
        "pop",
        "call",
        "ret",
    }

    header_re = re.compile(r"(.*):\s+file\s+format\s+(.*)")
    header_found = False
    kernel_format = None

    disassembly_re = re.compile(r"Disassembly\s+of\s+section\s+(.*):")
    sections = []

    symbol_re = re.compile(r"(\d+)\s+<(.*)>:")
    symbol_found = False

    instruction_re = re.compile(r"\s+([0-9a-z]+):\s+(\S+)\s+.*")
    total_count = 0
    arithmetic_count = 0
    move_count = 0
    packing_count = 0

    # ==
    for line in output.splitlines():

        if not line or line.isspace():
            continue

        elif m := header_re.match(line):
            header_found = True
            kernel_format = m.group(2)

        elif m := disassembly_re.match(line):
            sections.append(m.group(1))

        elif m := symbol_re.match(line):
            if (vectorized and m.group(2) == "vec_task_0") or (not vectorized and m.group(2) == "task_0"):
                symbol_found = True

        elif m := instruction_re.match(line):
            mnemonic = m.group(2)
            if mnemonic in arithmetic_insts:
                arithmetic_count = arithmetic_count + 1
            elif mnemonic in move_insts:
                move_count = move_count + 1
            elif mnemonic in vector_packing_insts:
                packing_count = packing_count + 1
            elif mnemonic not in other_insts:
                print(f"Unknown op: {mnemonic}")
                raise RuntimeError("Objdump failed: unknown mnemonic")
            total_count = total_count + 1

        else:
            print(line)
            raise RuntimeError("objdump failed: unexpected line")

    if not header_found:
        raise RuntimeError("objdump failed: no header found")
    elif not symbol_found:
        raise RuntimeError("objdump failed: requested symbol not found")

    data = {
        "kernel format": kernel_format,
        "#ops": total_count,
        "#arithmetic ops": arithmetic_count,
        "#move ops": move_count,
        "#packing ops": packing_count
    }
    return data


def invokeParse(kernel, vectorized):
    command = ["objdump", "-M", "intel", "--no-show-raw-insn"]
    if vectorized:
        command.append("--disassemble=vec_task_0")
    else:
        command.append("--disassemble=task_0")
    command.append(kernel[1])
    run_result = subprocess.run(command, capture_output=True, text=True)
    if run_result.returncode == 0:
        return parse_output(run_result.stdout, vectorized)
    else:
        print(f"Parsing of {kernel[0]} failed with error code {run_result.returncode}")
        print(run_result.stdout)
        print(run_result.stderr)
    return None


def get_kernels(kernelDir: str):
    kernels = []
    for file in os.listdir(kernelDir):
        baseName = os.path.basename(file)
        extension = os.path.splitext(baseName)[-1].lower()
        kernelName = os.path.splitext(baseName)[0]
        if extension == ".so":
            kernelFile = os.path.join(kernelDir, file)
            kernels.append((kernelName, kernelFile))
    print(f"Number of kernels found: {len(kernels)}")
    return kernels


def traverseKernels(kernelDir: str, vectorized: bool, logDir: str):
    kernels = get_kernels(kernelDir)

    # Sort kernels s.t. smaller ones appear first
    kernels = sorted(kernels, key=lambda m: os.path.getsize(m[1]))

    counter = 0
    for kernel in kernels:
        counter = counter + 1
        print(f"Current kernel ({counter}/{len(kernels)}): {kernel[0]}")
        data = {"Name": kernel[0]}
        data.update(invokeParse(kernel, vectorized))
        log_file_all = os.path.join(logDir, "kernel_data.csv")
        file_exists = os.path.isfile(log_file_all)
        if not os.path.isdir(logDir):
            os.makedirs(logDir)
        with open(log_file_all, 'a') as log_file:
            writer = csv.DictWriter(log_file, delimiter=",", fieldnames=data.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(data)


if __name__ == '__main__':
    fire.Fire(traverseKernels)
