import argparse
import os

from colorama import Fore, Back, Style

import copy

from riscvmodel.isa import Instruction
from riscvmodel.code import decode, MachineDecodeError
from bitstring import BitArray

MemSize = 1000 # memory size, in reality, the memory size should be 2^32, but for this lab, for the space resaon, we keep it as this large number, but the memory is still 32-bit addressable.

class InsMem(object):

    def __init__(self, name, ioDir):
        self.id = name

        with open(ioDir + "/imem.txt") as im:
            self.IMem = [data.replace("\n", "") for data in im.readlines()]

    def read_instr(self, read_address):
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

    def read_data(self, read_address):
        # read data memory
        # return 32-bit signed int value

        #Handle word addressing - use nearest lower multiple for 4 for address = x - x % 4
        read_address = read_address - read_address % 4
        if len(self.DMem) < read_address + 4:
            raise Exception("Data MEM - Out of bound access")
        return BitArray(bin="".join(self.DMem[read_address: read_address + 4])).int32

    def write_data_mem(self, address, write_data):
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

    def read_rf(self, reg_addr):
        return self.Registers[reg_addr]

    def write_rf(self, reg_addr, wrt_reg_data):
        if reg_addr != 0:
            self.Registers[reg_addr] = wrt_reg_data

    def output_rf(self, cycle):
        op = ["State of RF after executing cycle:\t" + str(cycle) + "\n"]
        op.extend(['{:032b}'.format(val & 0xffffffff) + "\n" for val in self.Registers])
        if cycle == 0:perm = "w"
        else: perm = "a"
        with open(self.outputFile, perm) as file:
            file.writelines(op)

class State(object):
    def __init__(self):
        self.IF = {"nop": False, "instruction_count":0, "PC":0, "instruction_ob": None}
        self.ID = {"nop": True, "Instr": 0, "instruction_ob": None}
        self.EX = {"nop": True, "Read_data1": 0, "Read_data2": 0, "Rd":0, 
                   "Store_data":0, "Imm": 0, "Rs": 0, "Rt": 0, "Wrt_reg_addr": 0, 
                   "is_I_type": False, "rd_mem": 0, 
                   "wrt_mem": 0, "opcode": 0, "wrt_enable": 0, "instruction_ob": None}
        self.MEM = {"nop": True, "ALUresult": 0, "Store_data": 0, "Rs": 0, 
                    "Rt": 0, "Wrt_reg_addr": 0, "rd_mem": 0, 
                   "wrt_mem": 0, "wrt_enable": 0, "instruction_ob": None}
        self.WB = {"nop": True, "Wrt_data": 0, "Rs": 0, "Rt": 0, "Wrt_reg_addr": 0, 
                   "wrt_enable": 0, "instruction_ob": None}

    def __str__(self):
        # DONE: update __str__ to make use of individual State objects
        return "\n\n".join([str(self.IF), str(self.ID), str(self.EX), str(self.MEM), str(self.WB)])


# class InstructionBase():

#     def __init__(self, instruction: Instruction, memory: DataMem, registers: RegisterFile, state: State,
#                  nextState: State):
#         self.instruction = instruction
#         self.memory = memory
#         self.registers = registers
#         self.state = state
#         self.nextState = nextState

#     def mem_ss(self, *args, **kwargs):
#         pass

#     def wb_ss(self, *args, **kwargs):
#         pass

#     def execute(self, *args, **kwargs):
#         return self.execute_ss(*args, **kwargs)

#     def mem(self, *args, **kwargs):
#         return self.mem_ss(*args, **kwargs)

#     def wb_ss(self, *args, **kwargs):
#         data = kwargs['alu_result']
#         return self.registers.write_rf(self.rd, data)

# class InstructionRBase(InstructionBase):
#     def __init__(self, instruction: Instruction, memory: DataMem, registers: RegisterFile, state: State,
#                  nextState: State):
#         super(InstructionRBase, self).__init__(instruction, memory, registers, state, nextState)
#         self.rs1 = instruction.rs1
#         self.rs2 = instruction.rs2
#         self.rd = instruction.rd

#     # def wb_ss(self, *args, **kwargs):
#     #     data = kwargs['alu_result']
#     #     return self.registers.write_rf(self.rd, data)

# class InstructionIBase(InstructionBase):
#     def __init__(self, instruction: Instruction, memory: DataMem, registers: RegisterFile, state: State,
#                  nextState: State):
#         super(InstructionIBase, self).__init__(instruction, memory, registers, state, nextState)
#         self.rs1 = instruction.rs1
#         self.rd = instruction.rd
#         self.imm = instruction.imm.value

#     # def wb_ss(self, *args, **kwargs):
#     #     data = kwargs['alu_result']
#     #     return self.registers.write_rf(self.rd, data)

# class InstructionSBase(InstructionBase):
#     def __init__(self, instruction: Instruction, memory: DataMem, registers: RegisterFile, state: State,
#                  nextState: State):
#         super(InstructionSBase, self).__init__(instruction, memory, registers, state, nextState)
#         self.rs1 = instruction.rs1
#         self.rs2 = instruction.rs2
#         self.imm = instruction.imm.value

#     def mem_ss(self, *args, **kwargs):
#         address = kwargs['alu_result']
#         data = self.registers.read_rf(self.rs2)
#         self.memory.write_data_mem(address, data)

# class InstructionBBase(InstructionBase):
#     def __init__(self, instruction: Instruction, memory: DataMem, registers: RegisterFile, state: State,
#                  nextState: State):
#         super(InstructionBBase, self).__init__(instruction, memory, registers, state, nextState)
#         self.rs1 = instruction.rs1
#         self.rs2 = instruction.rs2
#         self.imm = instruction.imm.value

#     def take_branch(self, operand1, operand2):
#         pass

#     def execute_ss(self, *args, **kwargs):
#         pass

# class InstructionJBase(InstructionBase):
#     def __init__(self, instruction: Instruction, memory: DataMem, registers: RegisterFile, state: State,
#                  nextState: State):
#         super(InstructionJBase, self).__init__(instruction, memory, registers, state, nextState)
#         self.rd = instruction.rd
#         self.imm = instruction.imm.value

#     def execute_ss(self, *args, **kwargs):
#         pass

# class ADD(InstructionRBase):
#     def __init__(self, instruction: Instruction, memory: DataMem, registers: RegisterFile, state: State,
#                  nextState: State):
#         super(ADD, self).__init__(instruction, memory, registers, state, nextState)

#     def execute_ss(self, *args, **kwargs):
#         return self.registers.read_rf(self.rs1) + self.registers.read_rf(self.rs2)

# class SUB(InstructionRBase):

#     def __init__(self, instruction: Instruction, memory: DataMem, registers: RegisterFile, state: State,
#                  nextState: State):
#         super(SUB, self).__init__(instruction, memory, registers, state, nextState)

#     def execute_ss(self, *args, **kwargs):
#         return self.registers.read_rf(self.rs1) - self.registers.read_rf(self.rs2)

# class XOR(InstructionRBase):

#     def __init__(self, instruction: Instruction, memory: DataMem, registers: RegisterFile, state: State,
#                  nextState: State):
#         super(XOR, self).__init__(instruction, memory, registers, state, nextState)

#     def execute_ss(self, *args, **kwargs):
#         return self.registers.read_rf(self.rs1) ^ self.registers.read_rf(self.rs2)

# class OR(InstructionRBase)  :

#     def __init__(self, instruction: Instruction, memory: DataMem, registers: RegisterFile, state: State,
#                  nextState: State):
#         super(OR, self).__init__(instruction, memory, registers, state, nextState)

#     def execute_ss(self, *args, **kwargs):
#         return self.registers.read_rf(self.rs1) | self.registers.read_rf(self.rs2)

# class AND(InstructionRBase):
#     def __init__(self, instruction: Instruction, memory: DataMem, registers: RegisterFile, state: State,
#                  nextState: State):
#         super(AND, self).__init__(instruction, memory, registers, state, nextState)

#     def execute_ss(self, *args, **kwargs):
#         return self.registers.read_rf(self.rs1) & self.registers.read_rf(self.rs2)

# class ADDI(InstructionIBase):
#     def __init__(self, instruction: Instruction, memory: DataMem, registers: RegisterFile, state: State,
#                  nextState: State):
#         super(ADDI, self).__init__(instruction, memory, registers, state, nextState)

#     def execute_ss(self, *args, **kwargs):
#         return self.registers.read_rf(self.rs1) + self.imm

# class XORI(InstructionIBase):
#     def __init__(self, instruction: Instruction, memory: DataMem, registers: RegisterFile, state: State,
#                  nextState: State):
#         super(XORI, self).__init__(instruction, memory, registers, state, nextState)

#     def execute_ss(self, *args, **kwargs):
#         return self.registers.read_rf(self.rs1) ^ self.imm

# class ORI(InstructionIBase):

#     def __init__(self, instruction: Instruction, memory: DataMem, registers: RegisterFile, state: State,
#                  nextState: State):
#         super(ORI, self).__init__(instruction, memory, registers, state, nextState)

#     def execute_ss(self, *args, **kwargs):
#         return self.registers.read_rf(self.rs1) | self.imm

# class ANDI(InstructionIBase):
#     def __init__(self, instruction: Instruction, memory: DataMem, registers: RegisterFile, state: State,
#                  nextState: State):
#         super(ANDI, self).__init__(instruction, memory, registers, state, nextState)

#     def execute_ss(self, *args, **kwargs):
#         return self.registers.read_rf(self.rs1) & self.imm

# class LW(InstructionIBase):
#     def __init__(self, instruction: Instruction, memory: DataMem, registers: RegisterFile, state: State,
#                  nextState: State):
#         super(LW, self).__init__(instruction, memory, registers, state, nextState)

#     def execute_ss(self, *args, **kwargs):
#         return self.registers.read_rf(self.rs1) + self.imm

#     def mem_ss(self, *args, **kwargs):
#         address = kwargs['alu_result']
#         return self.memory.read_data(address)

#     def wb_ss(self, *args, **kwargs):
#         data = kwargs['mem_result']
#         return self.registers.write_rf(self.rd, data)

# class SW(InstructionSBase):
#     def __init__(self, instruction: Instruction, memory: DataMem, registers: RegisterFile, state: State,
#                  nextState: State):
#         super(SW, self).__init__(instruction, memory, registers, state, nextState)

#     def execute_ss(self, *args, **kwargs):
#         return self.registers.read_rf(self.rs1) + self.imm


# class BEQ(InstructionBBase):

#     def __init__(self, instruction: Instruction, memory: DataMem, registers: RegisterFile, state: State,
#                  nextState: State):
#         super(BEQ, self).__init__(instruction, memory, registers, state, nextState)

#     def take_branch(self, operand1, operand2):
#         return operand1 == operand2


# class BNE(InstructionBBase):

#     def __init__(self, instruction: Instruction, memory: DataMem, registers: RegisterFile, state: State,
#                  nextState: State):
#         super(BNE, self).__init__(instruction, memory, registers, state, nextState)

#     def take_branch(self, operand1, operand2):
#         return operand1 != operand2


# class JAL(InstructionJBase):

#     def __init__(self, instruction: Instruction, memory: DataMem, registers: RegisterFile, state: State,
#                  nextState: State):
#         super(JAL, self).__init__(instruction, memory, registers, state, nextState)


# class ADDERBTYPE:
#     def __init__(self, instruction: Instruction, state: State(), registers: RegisterFile):
#         self.instruction = instruction
#         self.state = state
#         self.registers = registers
#         self.rs1 = instruction.rs1
#         self.rs2 = instruction.rs2
#         self.imm = instruction.imm.value

#     def get_pc(self, *args, **kwargs):
#         if self.instruction.mnemonic == 'beq':
#             if self.registers.read_rf(self.rs1) == self.registers.read_rf(self.rs2):
#                 return self.state.IF["PC"] + self.imm
#             else:
#                 return self.state.IF["PC"] + 4
#         else:
#             if self.registers.read_rf(self.rs1) != self.registers.read_rf(self.rs2):
#                 return self.state.IF["PC"] + self.imm
#             else:
#                 return self.state.IF["PC"] + 4


# class ADDERJTYPE:
#     def __init__(self, instruction: Instruction, state: State(), registers: RegisterFile):
#         self.instruction = instruction
#         self.state = state
#         self.registers = registers
#         self.rd = instruction.rd
#         self.imm = instruction.imm.value

#     def get_pc(self, *args, **kwargs):
#         self.registers.write_rf(self.rd, self.state.IF["PC"] + 4)
#         return self.state.IF["PC"] + self.imm

class InstructionBase():

    def __init__(self, instruction: Instruction, memory: DataMem, registers: RegisterFile, state: State,
                 nextState: State):
        self.instruction = instruction
        self.memory = memory
        self.registers = registers
        self.state = state
        self.nextState = nextState
        self.stages = self.memory.id

    def decode_ss(self, *args, **kwargs):
        pass

    def execute_ss(self, *args, **kwargs):
        pass

    def mem_ss(self, *args, **kwargs):
        pass

    def wb_ss(self, *args, **kwargs):
        pass

    def decode_fs(self, *args, **kwargs):
        pass

    def execute_fs(self, *args, **kwargs):
        pass

    def mem_fs(self, *args, **kwargs):

        wb_state = State().WB
        wb_state.update({
            'instruction_ob':self.state.MEM['instruction_ob'],
            'nop':self.state.MEM['nop'],
            'Wrt_data':self.state.MEM['Store_data'],
            'Wrt_reg_addr':self.state.MEM['Wrt_reg_addr'],
            'wrt_enable':self.state.MEM['wrt_enable'],
            # 'halt':self.state.MEM.halt
        }
        )

        self.nextState.WB = wb_state

    def wb_fs(self, *args, **kwargs):
        if self.state.WB['wrt_enable']:
            self.registers.write_rf(self.state.WB['Wrt_reg_addr'], self.state.WB['Wrt_data'])

    def decode(self, *args, **kwargs):
        if self.stages == "SS":
            return self.decode_ss(*args, **kwargs)
        else:
            self.state = kwargs["state"]
            self.nextState = kwargs["nextState"]
            self.memory = kwargs["memory"]
            self.registers = kwargs["registers"]
            return self.state, self.nextState, self.memory, self.registers, self.decode_fs(*args, **kwargs)

    def execute(self, *args, **kwargs):
        if self.stages == "SS":
            return self.execute_ss(*args, **kwargs)
        else:
            self.state = kwargs["state"]
            self.nextState = kwargs["nextState"]
            self.memory = kwargs["memory"]
            self.registers = kwargs["registers"]
            response = self.execute_fs(*args, **kwargs)
            return self.state, self.nextState, self.memory, self.registers, response

    def mem(self, *args, **kwargs):
        if self.stages == "SS":
            return self.mem_ss(*args, **kwargs)
        else:
            self.state = kwargs["state"]
            self.nextState = kwargs["nextState"]
            self.memory = kwargs["memory"]
            self.registers = kwargs["registers"]
            response = self.mem_fs(*args, **kwargs)
            return self.state, self.nextState, self.memory, self.registers, response

    def wb(self, *args, **kwargs):
        if self.stages == "SS":
            return self.wb_ss(*args, **kwargs)
        else:
            self.state = kwargs["state"]
            self.nextState = kwargs["nextState"]
            self.memory = kwargs["memory"]
            self.registers = kwargs["registers"]
            response = self.wb_fs(*args, **kwargs)
            return self.state, self.nextState, self.memory, self.registers, response


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

    def decode_fs(self, *args, **kwargs):
        ex_state = State().EX

        # TODO: Handle Hazards
        #   set nop for EX state
        #   will be applicable in R, I, S, B, J type instructions

        ex_state.update({
            'instruction_ob':self,
            'nop' :self.state.ID['nop'],
            'Read_data1' :self.registers.read_rf(self.rs1),
            'Read_data2' :self.registers.read_rf(self.rs2),
            'Rd' :self.rd,
            'wrt_enable' :True
        })

        # Stall
        if self.state.EX['Rd'] in [self.rs1, self.rs2] \
                and self.state.EX['rd_mem'] \
                and  self.rs1 != 0 and self.rs2 != 0:
            print("Stall")
            print(ex_state)
            ex_state['nop'] = True
            self.state.IF['PC'] -= 4
            self.nextState.EX = ex_state
            self.nextState.IF['instruction_count'] = self.nextState.IF['instruction_count'] - 1
            return

        # Forwarding
        if self.state.MEM['rd_mem'] and self.state.MEM['wrt_enable'] and \
            not self.state.MEM['wrt_mem'] and \
            self.state.MEM['Wrt_reg_addr'] == self.rs1 and self.rs1 != 0:
            print("Forwarding")
            print(ex_state)
            ex_state['Read_data1'] = self.nextState.WB['Wrt_data']

        if self.state.MEM['rd_mem'] and self.state.MEM['wrt_enable'] and \
            not self.state.MEM['wrt_mem'] and \
            self.state.MEM['Wrt_reg_addr'] == self.rs2 and self.rs2 != 0:
            print("Forwarding")
            print(ex_state)
            ex_state['Read_data2'] = self.nextState.WB['Wrt_data']

        if not self.state.EX['rd_mem'] and self.state.EX['wrt_enable'] and \
            not self.state.EX['wrt_mem'] and self.state.EX['Rd'] == self.rs1 and \
            self.rs1 != 0:
            print("Forwarding")
            print(ex_state)

            ex_state['Read_data1'] = self.nextState.MEM['Store_data']

        if not self.state.EX['rd_mem'] and self.state.EX['wrt_enable'] and \
            not self.state.EX['wrt_mem'] and self.state.EX['Rd'] == self.rs2 and \
            self.rs2 != 0:
            print("Forwarding")
            print(ex_state)

            ex_state['Read_data2'] = self.nextState.MEM['Store_data']

        self.nextState.EX = ex_state

    def execute_fs(self, *args, **kwargs):
        mem_state = State().MEM
        mem_state.update({
            'instruction_ob':self,
            'nop' :self.state.EX['nop'],
            'Wrt_reg_addr' :self.state.EX['Rd'],
            'wrt_enable' :True,
            # 'halt' :self.state.EX.halt
        })
        self.nextState.MEM = mem_state


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

    def decode_fs(self, *args, **kwargs):
        ex_state = State().EX
        ex_state.update({
            'instruction_ob':self,
            'nop' :self.state.ID['nop'],
            'Read_data1' :self.registers.read_rf(self.rs1),
            'Read_data2' :self.imm,
            'Rd' :self.rd,
            'wrt_enable' :True,
            # 'halt' :self.state.ID.halt
        })


        # Stall
        if self.state.EX['Rd'] == self.rs1 and self.state.EX['rd_mem'] and self.rs1 != 0:
            print("Stall")
            print(ex_state)
            ex_state['nop'] = True
            self.state.IF['PC'] -= 4
            self.nextState.EX = ex_state
            self.nextState.IF['instruction_count'] = self.nextState.IF['instruction_count'] - 1
            return

        # Forwarding
        if self.state.MEM['rd_mem'] and self.state.MEM['wrt_enable'] and \
            not self.state.MEM['wrt_mem'] and self.state.MEM['Wrt_reg_addr'] == self.rs1 \
            and self.rs1 != 0:
            print("Forwarding")
            print(ex_state)

            ex_state['Read_data1'] = self.nextState.WB['Wrt_data']

        if not self.state.EX['rd_mem'] and self.state.EX['wrt_enable'] and \
            not self.state.EX['wrt_mem'] and self.state.EX['Rd'] == self.rs1 and \
            self.rs1 != 0:
            print("Forwarding")
            print(ex_state)

            ex_state["Read_data1"] = self.nextState.MEM['Store_data']

        self.nextState.EX = ex_state

    def execute_fs(self, *args, **kwargs):
        # mem_state = MEMState()
        mem_state = State().MEM

        mem_state.update({
            'instruction_ob':self,
            'nop' :self.state.EX['nop'],
            'Wrt_reg_addr' :self.state.EX['Rd'],
            'wrt_enable' :True,
            # 'halt' :self.state.EX.halt
        })
        self.nextState.MEM = mem_state


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

    def decode_fs(self, *args, **kwargs):
        # ex_state = EXState()
        ex_state = State().EX

        ex_state.update({
            'instruction_ob':self,
            'nop' :self.state.ID['nop'],
            'Read_data1' :self.registers.read_rf(self.rs1),
            'Read_data2' :self.imm,
            'Store_data' :self.registers.read_rf(self.rs2),
            'Rd' :self.rs2,
            'wrt_mem' :True,
            #  halt=self.state.ID.halt
        })
        # Stall
        if self.state.EX['Rd'] in [self.rs1, self.rs2] and \
            self.state.EX['rd_mem'] and self.rs1 != 0 and self.rs2 != 0:
            ex_state['nop'] = True
            self.state.IF['PC'] -= 4
            self.nextState.EX = ex_state
            self.nextState.IF['instruction_count'] = self.nextState.IF['instruction_count'] - 1
            return

        # Forwarding
        if not self.state.EX['rd_mem'] and self.state.EX['wrt_enable'] and \
            not self.state.EX['wrt_mem'] and self.state.EX['Rd'] == self.rs1 \
            and self.rs1 != 0:
            print("Forwarding")
            print(ex_state)

            ex_state['Read_data1'] = self.nextState.MEM['Store_data']

        if not self.state.EX['rd_mem'] and self.state.EX['wrt_enable'] and \
            not self.state.EX['wrt_mem'] and self.state.EX['Rd'] == self.rs2 and \
            self.rs2 != 0:
            print("Forwarding")
            print(ex_state)

            ex_state['Store_data'] = self.nextState.MEM['Store_data']

        if not self.state.MEM['rd_mem'] and self.state.MEM['wrt_enable'] and \
            not self.state.MEM['wrt_mem'] and self.state.MEM['Wrt_reg_addr'] == self.rs1 and \
            self.rs1 != 0:
            print("Forwarding")
            print(ex_state)

            ex_state['Read_data1'] = self.nextState.WB['Wrt_data']

        if not self.state.MEM['rd_mem'] and self.state.MEM['wrt_enable'] and \
            not self.state.MEM['wrt_enable'] and self.state.MEM['Wrt_reg_addr'] == self.rs2 and \
            self.rs2 != 0:
            print("Forwarding")
            print(ex_state)

            ex_state['Store_data'] = self.nextState.WB['Wrt_data']

        self.nextState.EX = ex_state

    def execute_fs(self, *args, **kwargs):
        # mem_state = MEMState()
        mem_state = State().MEM
        mem_state.update({
            'instruction_ob':self,
            'nop' :self.state.EX['nop'],
            'data_address' :self.state.EX['Read_data1'] + self.state.EX['Read_data2'],
            'Store_data' :self.state.EX['Store_data'],
            'wrt_mem' :True,
            # halt=self.state.ID.halt
        })
        self.nextState.MEM = mem_state

    def mem_fs(self, *args, **kwargs):
        if self.state.MEM['wrt_mem']:
            self.memory.write_data_mem(self.state.MEM['data_address'], \
                                       self.state.MEM['Store_data'])
        # wb_state = WBState()
        wb_state = State().WB
        wb_state.update({
            'instruction_ob':self
        })
        self.nextState.WB = wb_state


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

    def execute_fs(self, *args, **kwargs):
        mem_state = State().MEM
        mem_state['instruction_ob'] = self
        mem_state['nop'] = True
        self.nextState.MEM = mem_state

    def decode_fs(self, *args, **kwargs):

        operand1 = self.registers.read_rf(self.rs1)
        operand2 = self.registers.read_rf(self.rs2)

        if self.state.EX['wrt_enable'] and self.state.EX['Rd'] != 0 and \
            self.state.EX['Rd'] == self.rs1 and self.rs1 != 0:
            operand1 = self.nextState.MEM['Store_data']

        if self.state.EX['wrt_enable'] and self.state.EX['Rd'] != 0 and \
            self.state.EX['Rd'] == self.rs2 and self.rs2 != 0:
            operand2 = self.nextState.MEM['Store_data']

        if self.state.MEM['wrt_enable'] and self.state.MEM['Wrt_reg_addr'] != 0 and \
            not ( self.state.EX['wrt_enable'] and self.state.EX['Rd'] != 0 and \
                self.state.EX['Rd'] == self.rs1) and \
                self.state.MEM['Wrt_reg_addr'] == self.rs1 and self.rs1 != 0:
            operand1 = self.nextState.WB['Store_data']

        if self.state.MEM['wrt_enable'] and self.state.MEM['Wrt_reg_addr'] != 0 and \
            not ( self.state.EX['wrt_enable'] and self.state.EX['Rd'] != 0 and \
                self.state.EX['Rd'] == self.rs2) and \
                self.state.MEM['Wrt_reg_addr'] == self.rs2 and self.rs2 != 0:
            operand2 = self.nextState.WB['Wrt_data']

        ex_state = State().EX
        ex_state['instruction_ob'] = self

        if self.take_branch(operand1, operand2):
            self.nextState.IF['PC'] = self.state.IF['PC'] + self.imm - 4
            self.nextState.ID['nop'] = True
            self.state.IF['nop'] = True
        ex_state['nop'] = True

        self.nextState.EX = ex_state


class InstructionJBase(InstructionBase):
    def __init__(self, instruction: Instruction, memory: DataMem, registers: RegisterFile, state: State,
                 nextState: State):
        super(InstructionJBase, self).__init__(instruction, memory, registers, state, nextState)
        self.rd = instruction.rd
        self.imm = instruction.imm.value

    def execute_ss(self, *args, **kwargs):
        pass

    def decode_fs(self, *args, **kwargs):
        # ex_state = EXState()
        ex_state = State().EX
        ex_state.update({
            'instruction_ob' :self,
            'Store_data' :self.state.IF['PC'],
            'Rd' :self.rd,
            'wrt_enable' :True
        })

        self.nextState.IF['PC'] = self.state.IF['PC'] + self.imm - 4
        self.nextState.ID['nop'] = True
        self.state.IF['nop'] = True

        self.nextState.EX = ex_state

    def execute_fs(self, *args, **kwargs):
        # mem_state = MEMState()
        mem_state = State().MEM
        mem_state.update({
            'instruction_ob':self,
            'Store_data' :self.state.EX['Store_data'],
            'Wrt_reg_addr' :self.rd,
            'wrt_enable' :True
        })
        self.nextState.MEM = mem_state


class ADD(InstructionRBase):
    def __init__(self, instruction: Instruction, memory: DataMem, registers: RegisterFile, state: State,
                 nextState: State):
        super(ADD, self).__init__(instruction, memory, registers, state, nextState)

    def execute_ss(self, *args, **kwargs):
        return self.registers.read_rf(self.rs1) + self.registers.read_rf(self.rs2)

    def execute_fs(self, *args, **kwargs):
        super(ADD, self).execute_fs()
        self.nextState.MEM['Store_data'] = self.state.EX['Read_data1'] + self.state.EX['Read_data2']


class SUB(InstructionRBase):

    def __init__(self, instruction: Instruction, memory: DataMem, registers: RegisterFile, state: State,
                 nextState: State):
        super(SUB, self).__init__(instruction, memory, registers, state, nextState)

    def execute_ss(self, *args, **kwargs):
        return self.registers.read_rf(self.rs1) - self.registers.read_rf(self.rs2)

    def execute_fs(self, *args, **kwargs):
        super(SUB, self).execute_fs()
        self.nextState.MEM['Store_data'] = self.state.EX['Read_data1'] - self.state.EX['Read_data2']


class XOR(InstructionRBase):

    def __init__(self, instruction: Instruction, memory: DataMem, registers: RegisterFile, state: State,
                 nextState: State):
        super(XOR, self).__init__(instruction, memory, registers, state, nextState)

    def execute_ss(self, *args, **kwargs):
        return self.registers.read_rf(self.rs1) ^ self.registers.read_rf(self.rs2)

    def execute_fs(self, *args, **kwargs):
        super(XOR, self).execute_fs()
        self.nextState.MEM['Store_data'] = self.state.EX['Read_data1'] ^ self.state.EX['Read_data2']


class OR(InstructionRBase):

    def __init__(self, instruction: Instruction, memory: DataMem, registers: RegisterFile, state: State,
                 nextState: State):
        super(OR, self).__init__(instruction, memory, registers, state, nextState)

    def execute_ss(self, *args, **kwargs):
        return self.registers.read_rf(self.rs1) | self.registers.read_rf(self.rs2)

    def execute_fs(self, *args, **kwargs):
        super(OR, self).execute_fs()
        self.nextState.MEM['Store_data'] = self.state.EX['Read_data1'] | self.state.EX['Read_data2']


class AND(InstructionRBase):
    def __init__(self, instruction: Instruction, memory: DataMem, registers: RegisterFile, state: State,
                 nextState: State):
        super(AND, self).__init__(instruction, memory, registers, state, nextState)

    def execute_ss(self, *args, **kwargs):
        return self.registers.read_rf(self.rs1) & self.registers.read_rf(self.rs2)

    def execute_fs(self, *args, **kwargs):
        super(AND, self).execute_fs()
        self.nextState.MEM['Store_data'] = self.state.EX['Read_data1'] & self.state.EX['Read_data2']


class ADDI(InstructionIBase):
    def __init__(self, instruction: Instruction, memory: DataMem, registers: RegisterFile, state: State,
                 nextState: State):
        super(ADDI, self).__init__(instruction, memory, registers, state, nextState)

    def execute_ss(self, *args, **kwargs):
        return self.registers.read_rf(self.rs1) + self.imm

    def execute_fs(self, *args, **kwargs):
        super(ADDI, self).execute_fs()
        self.nextState.MEM['Store_data'] = self.state.EX['Read_data1'] + self.state.EX['Read_data2']


class XORI(InstructionIBase):
    def __init__(self, instruction: Instruction, memory: DataMem, registers: RegisterFile, state: State,
                 nextState: State):
        super(XORI, self).__init__(instruction, memory, registers, state, nextState)

    def execute_ss(self, *args, **kwargs):
        return self.registers.read_rf(self.rs1) ^ self.imm

    def execute_fs(self, *args, **kwargs):
        super(XORI, self).execute_fs()
        self.nextState.MEM['Store_data'] = self.state.EX['Read_data1'] ^ self.state.EX['Read_data2']


class ORI(InstructionIBase):

    def __init__(self, instruction: Instruction, memory: DataMem, registers: RegisterFile, state: State,
                 nextState: State):
        super(ORI, self).__init__(instruction, memory, registers, state, nextState)

    def execute_ss(self, *args, **kwargs):
        return self.registers.read_rf(self.rs1) | self.imm

    def execute_fs(self, *args, **kwargs):
        super(ORI, self).execute_fs()
        self.nextState.MEM['Store_data'] = self.state.EX['Read_data1'] | self.state.EX['Read_data2']


class ANDI(InstructionIBase):
    def __init__(self, instruction: Instruction, memory: DataMem, registers: RegisterFile, state: State,
                 nextState: State):
        super(ANDI, self).__init__(instruction, memory, registers, state, nextState)

    def execute_ss(self, *args, **kwargs):
        return self.registers.read_rf(self.rs1) & self.imm

    def execute_fs(self, *args, **kwargs):
        super(ANDI, self).execute_fs()
        self.nextState.MEM['Store_data'] = self.state.EX['Read_data1'] & self.state.EX['Read_data2']


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

    def decode_fs(self, *args, **kwargs):
        super(LW, self).decode_fs()
        self.nextState.EX['rd_mem'] = True

    def execute_fs(self, *args, **kwargs):
        super(LW, self).execute_fs()
        self.nextState.MEM.update({
            'data_address': self.state.EX['Read_data1'] + self.state.EX['Read_data2'],
            'rd_mem' : True
        })

    def mem_fs(self, *args, **kwargs):
        super(LW, self).mem_fs(*args, **kwargs)
        if self.state.MEM['rd_mem']:
            self.nextState.WB['Wrt_data'] = self.memory.read_data(
                self.state.MEM['data_address']
            )


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

class Core(object):
    def __init__(self, ioDir: str, imem: InsMem, dmem: DataMem):
        self.myRF = RegisterFile(ioDir)
        self.cycle = 0
        self.halted = False
        self.ioDir = ioDir
        self.state = State()
        self.nextState = State()
        self.ext_imem: InsMem = imem
        self.ext_dmem: DataMem = dmem

    def calculate_performance_metrics(self):
        cpi = float(self.cycle) / self.state.IF['instruction_count']
        ipc = 1 / cpi

        result_format = f"{'-'*15} Core Performance Metrics {'-'*15}\n" \
                        f"Number of cycles taken: {self.cycle}\n" \
                        f"Cycles per instruction: {cpi}\n" \
                        f"Instructions per cycle: {ipc}\n"

        print(self.ioDir[:-3] + "PerformanceMetrics_Result.txt")
        with open(self.ioDir[:-3] + "PerformanceMetrics_Result.txt", 'w') as file:
            file.write(result_format)

class SingleStageCore(Core):
    def __init__(self, ioDir: str, imem: InsMem, dmem: DataMem):
        super(SingleStageCore, self).__init__(ioDir + "/SS_", imem, dmem)
        self.opFilePath = ioDir + "/StateResult_SS.txt"

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
        printstate =  ["-" * 70 + "\n", "State after executing cycle: " + str(cycle) + "\n"]

        printstate.append("IF.PC: " + str(state.IF['PC']) + "\n")
        printstate.append("IF.nop: " + str(state.IF['nop']) + "\n")

        if (cycle == 0):perm = "w"
        else: perm = "a"
        with open(self.opFilePath, perm) as wf:
            wf.writelines(printstate)

class FiveStageCore(Core):
    def __init__(self, ioDir, imem, dmem):
        super(FiveStageCore, self).__init__(ioDir + "/FS_", imem, dmem)
        self.opFilePath = ioDir + "/StateResult_FS.txt"
        self.stages = "Five Stage"

    def step(self):
        # Your implementation

        # if self.state.IF["nop"] and self.state.ID["nop"] and self.state.EX["nop"] and self.state.MEM["nop"] and self.state.WB["nop"]:
        #     self.halted = True

        # if not self.state.WB['nop']: 
        #     wb_value = self.state.WB['Write_data'] if self.state.WB['MemtoReg'] else self.state.WB['ALUoutput']
        #     if self.state.WB['RegWrite'] and self.state.WB['Rd'] != 0: 
        #         self.myRF.writeRF(self.state.WB['Rd'], wb_value)

        # if not self.state.MEM['nop']: 
        #     rs2_data_raw = self.state.MEM['Read_data2']
        #     MemWrite = self.state.MEM['MemWrite']
        #     MemRead = self.state.MEM['MemRead']
        #     ALU_output_raw = self.state.MEM['ALUoutput']
        #     lw_value = 0

        #     if MemWrite: 
        #         self.ext_dmem.writeDataMem(ALU_output_raw, rs2_data_raw)
        #     elif MemRead: 
        #         lw_value = self.ext_dmem.readDataMem(ALU_output_raw)

        #     self.nextState.WB['nop'] = False
        #     self.nextState.WB['Rs1'] = self.state.MEM['Rs1']
        #     self.nextState.WB['Rs2'] = self.state.MEM['Rs2']
        #     self.nextState.WB['Rd'] = self.state.MEM['Rd']
        #     self.nextState.WB['RegWrite'] = self.state.MEM['RegWrite']
        #     self.nextState.WB['MemtoReg'] = self.state.MEM['MemtoReg']
        #     self.nextState.WB['ALUoutput'] = self.state.MEM['ALUoutput']
        #     self.state.MEM['Load_data'] = lw_value
        #     self.nextState.WB['Write_data'] = lw_value
        #     self.nextState.WB['nop'] = False
        #     self.nextState.MEM['nop'] = True
        # else:
        #     self.nextState.WB['nop'] = True
        
        # if not self.state.EX['nop']: 
        #     forwarding = self.forward_units()
        #     forwardA, forwardB = forwarding

        #     ins = self.state.EX['Ins']
        #     ALUOp = self.state.EX['ALUOp']
        #     rs1_data_raw = self.state.EX['Read_data1']
        #     rs2_data_raw = self.state.EX['Read_data2']
        #     imm_raw = self.state.EX['Imm']
        #     func7 = self.state.EX['funct7']
        #     func3 = self.state.EX['funct3']
        #     opcode = self.state.EX['opcode']

        #     ALU_con = self.ALU_control(opcode, func7, func3, ALUOp)
          
        #     input1_raw = self.ExMuxA(rs1_data_raw, forwardA)
        #     inputB_raw = self.ExMuxB(rs2_data_raw, forwardB)
        #     input2_raw = self.EX_MUX_2(inputB_raw, imm_raw)

        #     ALU_output_raw = int_to_bitstr(getALUOutput(ALU_con, ins, input1_raw, input2_raw))

        #     self.state.EX['ALUoutput'] = ALU_output_raw
        #     self.nextState.MEM['nop'] = False
        #     self.nextState.MEM['ALUoutput'] = ALU_output_raw
        #     self.nextState.MEM['Read_data2'] = inputB_raw
        #     self.nextState.MEM['Rs1'] = self.state.EX['Rs1']
        #     self.nextState.MEM['Rs2'] = self.state.EX['Rs2']
        #     self.nextState.MEM['Rd'] = self.state.EX['Rd']
        #     self.nextState.MEM['MemRead'] = self.state.EX['MemRead']
        #     self.nextState.MEM['MemWrite'] = self.state.EX['MemWrite']
        #     self.nextState.MEM['RegWrite'] = self.state.EX['RegWrite']
        #     self.nextState.MEM['MemtoReg'] = self.state.EX['MemtoReg']
        #     self.nextState.MEM['nop'] = False
        #     self.nextState.EX['nop'] = True
        # else: 
        #     self.nextState.MEM['nop'] = True



        # # --------------------- ID stage ----------------------
        # if not self.state.ID['nop']:
        #     if not self.state.IF['Flush']: 
        #         instruction = decode(int(self.state.ID['Instr']),2)
        #         instruction_ob: InstructionBase = get_instruction_class(instruction.mnemonic)(instruction,
        #                                                                                       self.ext_dmem,
        #                                                                                       self.myRF,
        #                                                                                       self.state,
        #                                                                                       self.nextState)

        #         self.state, self.nextState, self.ext_dmem, self.myRF, _ = instruction_ob.decode(state=self.state,
        #                                                                                         nextState=self.nextState,
        #                                                                                         registers=self.myRF,
        #                                                                                         memory=self.ext_dmem)


        #         PC = self.state.ID['PC']
        #         instr = self.ext_imem.readInstr(PC)
        #         func7 = instr[:7]
        #         func3 = instr[17:20]
        #         opcode = instr[25:]
        #         rs1_raw = instr[12:17]
        #         rs2_raw = instr[7:12]
        #         rd_raw = instr[20:25]
        #         ins = ''
        #         type = ''
        #         if opcode == '0110011':
        #             if func3 == '000':
        #                 if func7 == '0000000':
        #                     ins = 'ADD'
        #                 elif func7 == '0100000':
        #                     ins = 'SUB'
        #             elif func3 == '100':
        #                 ins = 'XOR'
        #             elif func3 == '110':
        #                 ins = 'OR'
        #             elif func3 == '111':
        #                 ins = 'AND'
        #             type = 'R'
        #         elif opcode == '0010011' or opcode == '0000011':
        #             if func3 == '000': 
        #                 if opcode == '0010011': 
        #                     ins = 'ADDI'
        #                 elif opcode == '0000011': 
        #                     ins = 'LW'
        #             elif func3 == '100': 
        #                 ins =  'XORI'
        #             elif func3 == '110': 
        #                 ins =  'ORI'
        #             elif func3 == '111': 
        #                 ins = 'ANDI'
        #             type = 'I'
        #         elif opcode == '1101111':
        #             ins = 'JAL'
        #             type = 'J'
        #         elif opcode == '1100011':
        #             if func3 == '000': 
        #                 ins = 'BEQ'
        #             elif func3 == '001': 
        #                 ins = 'BNE'
        #             type = 'B'
        #         elif opcode == '0100011':
        #             ins = 'SW'
        #             type = 'S'
        #         elif opcode == '1111111':
        #             ins = 'HALT'
        #             type = 'H'
        #         imm_raw = getImm(instr, type)
        #         rs2 = int(rs2_raw, 2)
        #         rs1 = int(rs1_raw, 2)
        #         rd = int(rd_raw, 2)
        #         rs1_data_raw = self.myRF.readRF(rs1)
        #         rs2_data_raw = self.myRF.readRF(rs2)
        #         if type == 'J': 
        #             rs1_data_raw = int_to_bitstr(PC)
        #             rs2_data_raw = int_to_bitstr(4)

        #         self.state.ID['Rs1'] = rs1
        #         self.state.ID['Rs2'] = rs2
        #         self.state.ID['Rd'] = rd

        #         main = {
        #             'branch': 0,
        #             'MemRead': 0,
        #             'MemtoReg': 0,
        #             'ALUOp': 0,
        #             'MemWrite': 0,
        #             'ALUSrc': 0,
        #             'RegWrite': 0
        #         }
        #         if type == 'R':
        #             main["RegWrite"] = 1
        #             main["ALUOp"] = 0b10
        #         elif type == 'I':
        #             main["ALUSrc"] = 1
        #             main["RegWrite"] = 1
        #             main["ALUOp"] = 0b10
        #             if ins == 'LW':
        #                 main["MemRead"] = 1
        #                 main["MemtoReg"] = 1
        #         elif type == 'S':
        #             main["ALUSrc"] = 1
        #             main["MemWrite"] = 1
        #             main["ALUOp"] = 0b00
        #         elif type == 'B':
        #             main["branch"] = 1
        #             main["ALUOp"] = 0b01
        #         elif type == 'J':
        #             main["RegWrite"] = 1
        #             main["branch"] = 1
        #             main["ALUOp"] = 0b10
        #         PCWrite, IF_IDWrite = self.hdu()
        #         self.state.IF['PCWrite'] = PCWrite
        #         if not PCWrite: 
        #             main["branch"] = 0
        #             main["MemRead"] = 0
        #             main["MemtoReg"] = 0
        #             main["ALUOp"] = 0
        #             main["MemWrite"] = 0
        #             main["ALUSrc"] = 0
        #             main["RegWrite"] = 0

        #         jump = 1
        #         forwardA, forwardB = self.forward_branches()
        #         compare1_raw = self.id_mux1(rs1_data_raw, forwardA)
        #         compare2_raw = self.id_mux2(rs2_data_raw, forwardB)
        #         if ins == 'BEQ': 
        #             jump = compare1_raw == compare2_raw
        #         elif ins == 'BNE': 
        #             jump = compare1_raw != compare2_raw
        #         self.state.IF['PCSrc'] = main["branch"] and jump
        #         self.nextState.IF['PC'] = self.b_MUX(imm_raw, PCWrite)
                
        #         if type != 'H': 
        #             if type != 'B':
        #                 self.nextState.EX['nop'] = False
        #                 self.nextState.EX['Ins'] = ins
        #                 self.nextState.EX['Read_data1'] = rs1_data_raw
        #                 self.nextState.EX['Read_data2'] = rs2_data_raw
        #                 self.nextState.EX['Imm'] = imm_raw
        #                 self.nextState.EX['Rs1'] = rs1
        #                 self.nextState.EX['Rs2'] = rs2
        #                 self.nextState.EX['Rd'] = rd
        #                 self.nextState.EX['funct3'] = func3
        #                 self.nextState.EX['funct7'] = func7
        #                 self.nextState.EX['opcode'] = opcode
        #                 self.nextState.EX['Branch'] = main["branch"]
        #                 self.nextState.EX['MemRead'] = main['MemRead']
        #                 self.nextState.EX['MemtoReg'] = main["MemtoReg"]
        #                 self.nextState.EX['ALUOp'] = main["ALUOp"]
        #                 self.nextState.EX['MemWrite'] = main["MemWrite"]
        #                 self.nextState.EX['ALUSrc'] = main["ALUSrc"]
        #                 self.nextState.EX['RegWrite'] = main["RegWrite"]
        #             else: 
        #                 self.nextState.EX['nop'] = True

        #             if not IF_IDWrite: 
        #             #     continue
        #             #     # print('{}\t{}\tx{}\tx{}\tx{}\t{}'.format(self.cycle, ins, rd, rs1, rs2, bitstr_to_int(imm_raw)))
        #             # else: 
        #                 # print('{}\tNOP'.format(self.cycle))
        #                 self.nextState.IF['PC'] = self.state.IF['PC']
        #                 self.nextState.ID = self.state.ID

        #         else: 
        #             self.nextState.EX['nop'] = True
        #             self.nextState.ID['nop'] = True
        #             self.nextState.IF['nop'] = True
        #             # print('{}\tHALT'.format(self.cycle))
        #     else: 
        #         # NOP for branch taken
        #         self.nextState.IF['PC'] = self.state.IF['PC'] + 4
        #         self.nextState.ID = self.state.ID
        #         self.nextState.EX['nop'] = True
        #         # print('{}\tNOP'.format(self.cycle))
        # else: 
        #     self.nextState.EX['nop'] = True

        # # --------------------- IF stage ----------------------
        # if not self.state.IF['nop']: 
        #     instr = self.ext_imem.readInstr(self.state.IF['PC'])

        #     if self.state.IF['PCWrite']:
        #         self.nextState.ID['PC'] = self.state.IF['PC']

        #     if self.state.ID['nop'] == True: 
        #         self.nextState.IF['PC'] = self.state.IF['PC'] + 4
            
        #     if self.state.IF['PCWrite']: 
        #         self.nextState.ID['Instr'] = instr
        # else: 
        #     self.nextState.IF['nop'] = True
        #     self.nextState.ID['nop'] = True
        
        # self.myRF.outputRF(self.cycle) # dump RF
        # self.printState(self.nextState, self.cycle) # print states after executing cycle 0, cycle 1, cycle 2 ... 
        
        # self.state = self.nextState
        # self.nextState = State() #The end of the cycle and updates the current state with the values calculated in this cycle
        # self.cycle += 1


        # Old implementation
 
        # --------------------- WB stage ----------------------
        # print(self.state)
        # print(self.state)
        if not self.state.WB["nop"]:
            self.state, self.nextState, self.ext_dmem, self.myRF, _ = self.state.WB["instruction_ob"].wb(
                state=self.state,
                nextState=self.nextState,
                registers=self.myRF,
                memory=self.ext_dmem)

        # --------------------- MEM stage ---------------------
        if not self.state.MEM["nop"]:

            self.state, self.nextState, self.ext_dmem, self.myRF, _ = self.state.MEM["instruction_ob"].mem(
                state=self.state,
                nextState=self.nextState,
                registers=self.myRF,
                memory=self.ext_dmem)
        else:
            self.nextState.WB["nop"] = True

        # --------------------- EX stage ----------------------
        if not self.state.EX["nop"]:

            self.state, self.nextState, self.ext_dmem, self.myRF, _ = self.state.EX["instruction_ob"].execute(
                state=self.state, nextState=self.nextState, registers=self.myRF, memory=self.ext_dmem)
        else:
            self.nextState.MEM["nop"] = True

        # --------------------- ID stage ----------------------
        if not self.state.ID["nop"]:
            # print("Instruction bytes", self.state.ID['Instr'])
            try:
                instruction = decode(int(self.state.ID['Instr'], 2))
                print(instruction)
                #returns Instruction Class which can access attributes of the instruction base class
                self.instruction_ob = get_instruction_class(instruction.mnemonic)(instruction,
                                                                                self.ext_dmem,
                                                                                self.myRF,
                                                                                self.state,
                                                                                self.nextState)
            
                print(Fore.RED + f"Current State: {self.state}")
                print(Fore.GREEN + f"Next State: {self.nextState}")



                self.state, self.nextState, self.ext_dmem, self.myRF, _ = self.instruction_ob.decode(state=self.state,
                                                                                                nextState=self.nextState,
                                                                                                registers=self.myRF,
                                                                                                memory=self.ext_dmem)
                print(Fore.RED + Back.YELLOW + f"Current State: {self.state}")
                print(Fore.GREEN + Back.YELLOW+ f"Next State: {self.nextState}")


                # print(self.state, self.nextState, self.ext_dmem, self.myRF)
            except MachineDecodeError as e:
                if "{:08x}".format(e.word) == 'ffffffff':
                    self.nextState.ID["halt"] = True
                else:
                    raise Exception("Invalid Instruction to Decode")
        else:
            self.nextState.EX["nop"] = True

        # --------------------- IF stage ----------------------
        if not self.state.IF["nop"]:
            self.nextState.ID['Instr'] = self.ext_imem.read_instr(self.state.IF["PC"])
            self.nextState.ID["nop"] = False
            if self.nextState.ID['Instr'] == "1" * 32:
                self.nextState.ID["nop"] = True
                self.nextState.IF["nop"] = True
            else:
                self.nextState.IF["PC"] = self.state.IF["PC"] + 4
                # self.nextState.IF['instruction_count'] = self.nextState.IF['instruction_count'] + 1

        else:
            self.nextState.ID["nop"] = True

        if self.state.IF["nop"] and self.state.ID["nop"]  and self.state.EX["nop"] and self.state.MEM["nop"] and self.state.WB["nop"]:
            self.nextState.IF['instruction_count'] = self.state.IF['instruction_count'] + 1
            self.halted = True

        self.myRF.output_rf(self.cycle)  # dump RF
        self.printState(self.nextState, self.cycle)  # print states after executing cycle 0, cycle 1, cycle 2 ...

        self.state = copy.deepcopy(self.nextState)
        self.nextState = State()
        self.cycle += 1
        print(Style.RESET_ALL)


    def printState(self, state, cycle):
        print_state = "\n" + "-" * 70 + "\n" + "State after executing cycle: " + str(cycle) + "\n\n"
        print_state += str(state)

        if (cycle == 0):
            perm = "w"
        else:
            perm = "a"
        with open(self.opFilePath, perm) as wf:
            wf.write(print_state)



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
    fsCore = FiveStageCore(ioDir, imem, dmem_fs)

    while True:
        if not ssCore.halted:
            ssCore.step()

        if not fsCore.halted:
            fsCore.step()

        if ssCore.halted and fsCore.halted:
            break

    # dump SS and FS data mem.
    dmem_ss.output_data_mem()
    dmem_fs.output_data_mem()


    ssCore.calculate_performance_metrics()
    fsCore.calculate_performance_metrics()
