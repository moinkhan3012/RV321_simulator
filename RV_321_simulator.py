import argparse
import os

import copy

from riscvmodel.isa import Instruction
from riscvmodel.code import decode, MachineDecodeError
from riscvmodel.isa import Instruction
from bitstring import BitArray

MemSize = 1000
# TODO: set nop default to false and handle it in init for core class
class InsMem(object):

    def __init__(self, name, ioDir):
        self.id = name

        with open(ioDir + "/imem.txt") as im:
            self.IMem = [data.replace("\n", "") for data in im.readlines()]

    def read_instr(self, read_address: int):
        # DONE: Handle word addressing - use nearest lower multiple for 4 for address = x - x % 4
        read_address = read_address - read_address % 4
        if len(self.IMem) < read_address + 4:
            raise Exception("Instruction MEM - Out of bound access")
        return "".join(self.IMem[read_address: read_address + 4])


class DataMem(object):
    def __init__(self, name, ioDir):
        self.id = name
        self.ioDir = ioDir
        with open(ioDir + "\\dmem.txt") as dm:
            self.DMem = [data.replace("\n", "") for data in dm.readlines()]

    def read_data(self, read_address: int) -> int:
        # read data memory
        # return 32-bit signed int value

        #Handle word addressing - use nearest lower multiple for 4 for address = x - x % 4
        read_address = read_address - read_address % 4
        if len(self.DMem) < read_address + 4:
            raise Exception("Data MEM - Out of bound access")
        return BitArray(bin="".join(self.DMem[read_address: read_address + 4])).int32

    def write_data_mem(self, address: int, write_data: int):
        # write data into byte addressable memory
        # Assuming data as 32 bit signed integer

        # Converting from int to bin

        # Handle word addressing - use nearest lower multiple for 4 for address = x - x % 4
        address = address - address % 4
        write_data = '{:032b}'.format(write_data & 0xffffffff)

        left, right, zeroes = [], [], []

        if address <= len(self.DMem):
            left = self.DMem[:address]
        else:
            left = self.DMem
            zeroes = ["0" * 8] * (address - len(self.DMem))
        if address + 4 <= len(self.DMem):
            right = self.DMem[address + 4:]

        self.DMem = left + zeroes + [write_data[i: i + 8] for i in range(0, 32, 8)] + right

    def output_data_mem(self):
        res_path = self.ioDir + "/" + self.id + "_DMEMResult.txt"
        with open(res_path, "w") as rp:
            rp.writelines([str(data) + "\n" for data in self.DMem])
            rp.writelines([ '0'*8 + '\n' for _ in range(MemSize - len(self.DMem))])

class RegisterFile(object):
    def __init__(self, ioDir):
        self.outputFile = ioDir + "RFResult.txt"
        self.Registers = [0x0 for i in range(32)]

    def read_rf(self, reg_addr: int) -> int:
        return self.Registers[reg_addr]

    def write_rf(self, reg_addr: int, wrt_reg_data: int):
        if reg_addr != 0:
            self.Registers[reg_addr] = wrt_reg_data

    def output_rf(self, cycle):
        op = ["State of RF after executing cycle:\t" + str(cycle) + "\n"]
        op.extend(['{:032b}'.format(val & 0xffffffff) + "\n" for val in self.Registers])
        if cycle == 0:
            perm = "w"
        else:
            perm = "a"
        with open(self.outputFile, perm) as file:
            file.writelines(op)

class State(object):

    def __init__(self):
        self.IF = {"nop": False, "PC": 0, "instruction_count":0}
        self.ID = {"nop": False, "Instr": 0}
        self.EX = {"nop": False, "Read_data1": 0, "Read_data2": 0, "Imm": 0, "Rs": 0, "Rt": 0, "Wrt_reg_addr": 0, "is_I_type": False, "rd_mem": 0, 
                   "wrt_mem": 0, "opcode": 0, "wrt_enable": 0}
        self.MEM = {"nop": False, "ALUresult": 0, "Store_data": 0, "Rs": 0, "Rt": 0, "Wrt_reg_addr": 0, "rd_mem": 0, 
                   "wrt_mem": 0, "wrt_enable": 0}
        self.WB = {"nop": False, "Wrt_data": 0, "Rs": 0, "Rt": 0, "Wrt_reg_addr": 0, "wrt_enable": 0}

    def nop_init(self):
        self.IF['nop'] = False
        self.ID['nop'] = True
        self.EX['nop'] = True
        self.MEM['nop'] = True
        self.WB['nop'] = True

    def __str__(self):
        # DONE: update __str__ to make use of individual State objects
        return "\n\n".join([str(self.IF), str(self.ID), str(self.EX), str(self.MEM), str(self.WB)])

class InstructionBase():

    def __init__(self, instruction: Instruction, memory: DataMem, registers: RegisterFile, state: State,
                 nextState: State):
        self.instruction = instruction
        self.memory = memory
        self.registers = registers
        self.state = state
        self.nextState = nextState

    def mem_ss(self, *args, **kwargs):
        pass

    def wb_ss(self, *args, **kwargs):
        pass

    def execute(self, *args, **kwargs):
        return self.execute_ss(*args, **kwargs)

    def mem(self, *args, **kwargs):
        return self.mem_ss(*args, **kwargs)

    def wb(self, *args, **kwargs):
        return self.wb_ss(*args, **kwargs)

class InstructionRBase(InstructionBase):
    def __init__(self, instruction: Instruction, memory: DataMem, registers: RegisterFile, state: State,
                 nextState: State):
        super(InstructionRBase, self).__init__(instruction, memory, registers, state, nextState)
        self.rs1 = instruction.rs1
        self.rs2 = instruction.rs2
        self.rd = instruction.rd

    def wb_ss(self, *args, **kwargs):
        data = kwargs['alu_result']
        return self.registers.write_rf(self.rd, data)

class InstructionIBase(InstructionBase):
    def __init__(self, instruction: Instruction, memory: DataMem, registers: RegisterFile, state: State,
                 nextState: State):
        super(InstructionIBase, self).__init__(instruction, memory, registers, state, nextState)
        self.rs1 = instruction.rs1
        self.rd = instruction.rd
        self.imm = instruction.imm.value

    def wb_ss(self, *args, **kwargs):
        data = kwargs['alu_result']
        return self.registers.write_rf(self.rd, data)

class InstructionSBase(InstructionBase):
    def __init__(self, instruction: Instruction, memory: DataMem, registers: RegisterFile, state: State,
                 nextState: State):
        super(InstructionSBase, self).__init__(instruction, memory, registers, state, nextState)
        self.rs1 = instruction.rs1
        self.rs2 = instruction.rs2
        self.imm = instruction.imm.value

    def mem_ss(self, *args, **kwargs):
        address = kwargs['alu_result']
        data = self.registers.read_rf(self.rs2)
        self.memory.write_data_mem(address, data)

class InstructionBBase(InstructionBase):
    def __init__(self, instruction: Instruction, memory: DataMem, registers: RegisterFile, state: State,
                 nextState: State):
        super(InstructionBBase, self).__init__(instruction, memory, registers, state, nextState)
        self.rs1 = instruction.rs1
        self.rs2 = instruction.rs2
        self.imm = instruction.imm.value

    def take_branch(self, operand1, operand2):
        pass

    def execute_ss(self, *args, **kwargs):
        pass

class InstructionJBase(InstructionBase):
    def __init__(self, instruction: Instruction, memory: DataMem, registers: RegisterFile, state: State,
                 nextState: State):
        super(InstructionJBase, self).__init__(instruction, memory, registers, state, nextState)
        self.rd = instruction.rd
        self.imm = instruction.imm.value

    def execute_ss(self, *args, **kwargs):
        pass

class ADD(InstructionRBase):
    def __init__(self, instruction: Instruction, memory: DataMem, registers: RegisterFile, state: State,
                 nextState: State):
        super(ADD, self).__init__(instruction, memory, registers, state, nextState)

    def execute_ss(self, *args, **kwargs):
        return self.registers.read_rf(self.rs1) + self.registers.read_rf(self.rs2)

class SUB(InstructionRBase):

    def __init__(self, instruction: Instruction, memory: DataMem, registers: RegisterFile, state: State,
                 nextState: State):
        super(SUB, self).__init__(instruction, memory, registers, state, nextState)

    def execute_ss(self, *args, **kwargs):
        return self.registers.read_rf(self.rs1) - self.registers.read_rf(self.rs2)

class XOR(InstructionRBase):

    def __init__(self, instruction: Instruction, memory: DataMem, registers: RegisterFile, state: State,
                 nextState: State):
        super(XOR, self).__init__(instruction, memory, registers, state, nextState)

    def execute_ss(self, *args, **kwargs):
        return self.registers.read_rf(self.rs1) ^ self.registers.read_rf(self.rs2)

class OR(InstructionRBase):

    def __init__(self, instruction: Instruction, memory: DataMem, registers: RegisterFile, state: State,
                 nextState: State):
        super(OR, self).__init__(instruction, memory, registers, state, nextState)

    def execute_ss(self, *args, **kwargs):
        return self.registers.read_rf(self.rs1) | self.registers.read_rf(self.rs2)

class AND(InstructionRBase):
    def __init__(self, instruction: Instruction, memory: DataMem, registers: RegisterFile, state: State,
                 nextState: State):
        super(AND, self).__init__(instruction, memory, registers, state, nextState)

    def execute_ss(self, *args, **kwargs):
        return self.registers.read_rf(self.rs1) & self.registers.read_rf(self.rs2)

class ADDI(InstructionIBase):
    def __init__(self, instruction: Instruction, memory: DataMem, registers: RegisterFile, state: State,
                 nextState: State):
        super(ADDI, self).__init__(instruction, memory, registers, state, nextState)

    def execute_ss(self, *args, **kwargs):
        return self.registers.read_rf(self.rs1) + self.imm

class XORI(InstructionIBase):
    def __init__(self, instruction: Instruction, memory: DataMem, registers: RegisterFile, state: State,
                 nextState: State):
        super(XORI, self).__init__(instruction, memory, registers, state, nextState)

    def execute_ss(self, *args, **kwargs):
        return self.registers.read_rf(self.rs1) ^ self.imm

class ORI(InstructionIBase):

    def __init__(self, instruction: Instruction, memory: DataMem, registers: RegisterFile, state: State,
                 nextState: State):
        super(ORI, self).__init__(instruction, memory, registers, state, nextState)

    def execute_ss(self, *args, **kwargs):
        return self.registers.read_rf(self.rs1) | self.imm

class ANDI(InstructionIBase):
    def __init__(self, instruction: Instruction, memory: DataMem, registers: RegisterFile, state: State,
                 nextState: State):
        super(ANDI, self).__init__(instruction, memory, registers, state, nextState)

    def execute_ss(self, *args, **kwargs):
        return self.registers.read_rf(self.rs1) & self.imm

class LW(InstructionIBase):
    def __init__(self, instruction: Instruction, memory: DataMem, registers: RegisterFile, state: State,
                 nextState: State):
        super(LW, self).__init__(instruction, memory, registers, state, nextState)

    def execute_ss(self, *args, **kwargs):
        return self.registers.read_rf(self.rs1) + self.imm

    def mem_ss(self, *args, **kwargs):
        address = kwargs['alu_result']
        return self.memory.read_data(address)

    def wb_ss(self, *args, **kwargs):
        data = kwargs['mem_result']
        return self.registers.write_rf(self.rd, data)

class SW(InstructionSBase):
    def __init__(self, instruction: Instruction, memory: DataMem, registers: RegisterFile, state: State,
                 nextState: State):
        super(SW, self).__init__(instruction, memory, registers, state, nextState)

    def execute_ss(self, *args, **kwargs):
        return self.registers.read_rf(self.rs1) + self.imm


class BEQ(InstructionBBase):

    def __init__(self, instruction: Instruction, memory: DataMem, registers: RegisterFile, state: State,
                 nextState: State):
        super(BEQ, self).__init__(instruction, memory, registers, state, nextState)

    def take_branch(self, operand1, operand2):
        return operand1 == operand2


class BNE(InstructionBBase):

    def __init__(self, instruction: Instruction, memory: DataMem, registers: RegisterFile, state: State,
                 nextState: State):
        super(BNE, self).__init__(instruction, memory, registers, state, nextState)

    def take_branch(self, operand1, operand2):
        return operand1 != operand2


class JAL(InstructionJBase):

    def __init__(self, instruction: Instruction, memory: DataMem, registers: RegisterFile, state: State,
                 nextState: State):
        super(JAL, self).__init__(instruction, memory, registers, state, nextState)


class ADDERBTYPE:
    def __init__(self, instruction: Instruction, state: State(), registers: RegisterFile):
        self.instruction = instruction
        self.state = state
        self.registers = registers
        self.rs1 = instruction.rs1
        self.rs2 = instruction.rs2
        self.imm = instruction.imm.value

    def get_pc(self, *args, **kwargs):
        if self.instruction.mnemonic == 'beq':
            if self.registers.read_rf(self.rs1) == self.registers.read_rf(self.rs2):
                return self.state.IF['PC'] + self.imm
            else:
                return self.state.IF['PC'] + 4
        else:
            if self.registers.read_rf(self.rs1) != self.registers.read_rf(self.rs2):
                return self.state.IF['PC'] + self.imm
            else:
                return self.state.IF['PC'] + 4


class ADDERJTYPE:
    def __init__(self, instruction: Instruction, state: State(), registers: RegisterFile):
        self.instruction = instruction
        self.state = state
        self.registers = registers
        self.rd = instruction.rd
        self.imm = instruction.imm.value

    def get_pc(self, *args, **kwargs):
        self.registers.write_rf(self.rd, self.state.IF['PC'] + 4)
        return self.state.IF['PC'] + self.imm

def get_instruction_class(mnemonic):
    try:
        if mnemonic == "lb":
            mnemonic = "lw"
        
        cls = eval(mnemonic.upper())
        return cls
    except AttributeError as e:
        raise Exception("Invalid Instruction")
    

#rv32i

# memory size, in reality, the memory size should be 2^32, but for this lab, for the space reason
# we keep it as this large number, but the memory is still 32-bit addressable.

class Core(object):
    def __init__(self, ioDir: str, imem: InsMem, dmem: DataMem):
        self.myRF = RegisterFile(ioDir)
        self.cycle = 0
        self.halted = False
        self.ioDir = ioDir
        self.state = State()
        self.state.nop_init()
        self.nextState = State()
        self.nextState.nop_init()
        self.ext_imem: InsMem = imem
        self.ext_dmem: DataMem = dmem

    def calculate_performance_metrics(self):
        cpi = float(self.cycle) / self.state.IF['instruction_count']
        ipc = 1 / cpi

        result_format = f"{self.stages} Core Performance Metrics-----------------------------\n" \
                        f"Number of cycles taken: {self.cycle}\n" \
                        f"Cycles per instruction: {cpi}\n" \
                        f"Instructions per cycle: {ipc}\n"

        write_mode = "w" if self.stages == "Single Stage" else "a"

        print(self.ioDir[:-3] + "PerformanceMetrics_Result.txt")
        with open(self.ioDir[:-3] + "PerformanceMetrics_Result.txt", write_mode) as file:
            file.write(result_format)

class SingleStageCore(Core):
    def __init__(self, ioDir: str, imem: InsMem, dmem: DataMem):
        super(SingleStageCore, self).__init__(ioDir + "/SS_", imem, dmem)
        self.opFilePath = ioDir + "/StateResult_SS.txt"
        self.stages = "Single Stage"

    def step(self):
        # IF
        instruction_bytes = self.ext_imem.read_instr(self.state.IF['PC'])
        self.nextState.IF['instruction_count'] = self.nextState.IF['instruction_count'] + 1

        if instruction_bytes == "1" * 32:
            self.nextState.IF['nop'] = True
        else:
            self.nextState.IF['PC'] += 4

        try:
            # ID
            instruction: Instruction = decode(int(instruction_bytes, 2))

            if instruction.mnemonic in ['beq', 'bne']:
                self.nextState.IF['PC'] = ADDERBTYPE(instruction, self.state, self.myRF).get_pc()
            elif instruction.mnemonic == 'jal':
                self.nextState.IF['PC'] = ADDERJTYPE(instruction, self.state, self.myRF).get_pc()
            else:
                instruction_ob: InstructionBase = get_instruction_class(instruction.mnemonic)(instruction,
                                                                                              self.ext_dmem, self.myRF,
                                                                                              self.state,
                                                                                              self.nextState)
                # Ex
                alu_result = instruction_ob.execute()
                # Load/Store (MEM)
                mem_result = instruction_ob.mem(alu_result=alu_result)
                # WB
                wb_result = instruction_ob.wb(mem_result=mem_result, alu_result=alu_result)
        except MachineDecodeError as e:
            if "{:08x}".format(e.word) == 'ffffffff':
                pass
            else:
                raise Exception("Invalid Instruction to Decode")
        # self.halted = True
        if self.state.IF['nop']:
            self.halted = True

        self.myRF.output_rf(self.cycle)  # dump RF
        self.printState(self.nextState, self.cycle)  # print states after executing cycle 0, cycle 1, cycle 2 ...

        # The end of the cycle and updates the current state with the values calculated in this cycle
        self.state = copy.deepcopy(self.nextState)
        self.cycle += 1

    def printState(self, state, cycle):
        printstate = ["-" * 70 + "\n", "State after executing cycle: " + str(cycle) + "\n"]
        printstate.append("IF.PC: " + str(state.IF['PC']) + "\n")
        printstate.append("IF.nop: " + str(state.IF['nop']) + "\n")

        if (cycle == 0):
            perm = "w"
        else:
            perm = "a"
        with open(self.opFilePath, perm) as wf:
            wf.writelines(printstate)

if __name__ == "__main__":
    # parse arguments for input file location
    parser = argparse.ArgumentParser(description='RV32I processor')
    parser.add_argument('--iodir', default="", type=str, help='Directory containing the input files.')
    args = parser.parse_args()
    test_case_number = 1

    ioDir = os.path.abspath(args.iodir)

    print("IO Directory:", ioDir)

    imem = InsMem("Imem", ioDir)
    dmem_ss = DataMem("SS", ioDir)
    dmem_fs = DataMem("FS", ioDir)

    ssCore = SingleStageCore(ioDir, imem, dmem_ss)

    while True:
        if not ssCore.halted:
            ssCore.step()

        if ssCore.halted:# and fsCore.halted:
            break

    # dump SS and FS data mem.
    dmem_ss.output_data_mem()
    dmem_fs.output_data_mem()

    ssCore.calculate_performance_metrics()