# automatically generated by the FlatBuffers compiler, do not modify

# namespace: fbs

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class OptimizerGroup(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = OptimizerGroup()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsOptimizerGroup(cls, buf, offset=0):
        """This method is deprecated. Please switch to GetRootAs."""
        return cls.GetRootAs(buf, offset)
    @classmethod
    def OptimizerGroupBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b"\x4F\x44\x54\x43", size_prefixed=size_prefixed)

    # OptimizerGroup
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # OptimizerGroup
    def GroupName(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.String(o + self._tab.Pos)
        return None

    # OptimizerGroup
    def Step(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int64Flags, o + self._tab.Pos)
        return 0

    # OptimizerGroup
    def InitialLearningRate(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Float32Flags, o + self._tab.Pos)
        return 0.0

    # OptimizerGroup
    def OptimizerStates(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            x = self._tab.Vector(o)
            x += flatbuffers.number_types.UOffsetTFlags.py_type(j) * 4
            x = self._tab.Indirect(x)
            from ort_flatbuffers_py.fbs.ParameterOptimizerState import ParameterOptimizerState
            obj = ParameterOptimizerState()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # OptimizerGroup
    def OptimizerStatesLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # OptimizerGroup
    def OptimizerStatesIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        return o == 0

def OptimizerGroupStart(builder):
    builder.StartObject(4)

def Start(builder):
    OptimizerGroupStart(builder)

def OptimizerGroupAddGroupName(builder, groupName):
    builder.PrependUOffsetTRelativeSlot(0, flatbuffers.number_types.UOffsetTFlags.py_type(groupName), 0)

def AddGroupName(builder, groupName):
    OptimizerGroupAddGroupName(builder, groupName)

def OptimizerGroupAddStep(builder, step):
    builder.PrependInt64Slot(1, step, 0)

def AddStep(builder, step):
    OptimizerGroupAddStep(builder, step)

def OptimizerGroupAddInitialLearningRate(builder, initialLearningRate):
    builder.PrependFloat32Slot(2, initialLearningRate, 0.0)

def AddInitialLearningRate(builder, initialLearningRate):
    OptimizerGroupAddInitialLearningRate(builder, initialLearningRate)

def OptimizerGroupAddOptimizerStates(builder, optimizerStates):
    builder.PrependUOffsetTRelativeSlot(3, flatbuffers.number_types.UOffsetTFlags.py_type(optimizerStates), 0)

def AddOptimizerStates(builder, optimizerStates):
    OptimizerGroupAddOptimizerStates(builder, optimizerStates)

def OptimizerGroupStartOptimizerStatesVector(builder, numElems):
    return builder.StartVector(4, numElems, 4)

def StartOptimizerStatesVector(builder, numElems: int) -> int:
    return OptimizerGroupStartOptimizerStatesVector(builder, numElems)

def OptimizerGroupEnd(builder):
    return builder.EndObject()

def End(builder):
    return OptimizerGroupEnd(builder)
