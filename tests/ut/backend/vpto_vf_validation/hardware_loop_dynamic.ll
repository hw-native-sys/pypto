; ModuleID = 'ptoas.hivm.official.vector'
source_filename = "ptoas.hivm.official.vector"

declare <256 x i1> @llvm.hivm.pset.b32(i32)

declare <64 x float> @llvm.hivm.vldsx1.v64f32(ptr addrspace(6), i32, i32, i32)

declare <64 x float> @llvm.hivm.vadd.s.x.v64f32(<64 x float>, <64 x float>, <256 x i1>)

declare <64 x float> @llvm.hivm.vmul.s.x.v64f32(<64 x float>, <64 x float>, <256 x i1>)

declare void @llvm.hivm.vstsx1.v64f32(<64 x float>, ptr addrspace(6), i32, i32, i32, <256 x i1>)

declare void @llvm.hivm.mem.bar.vv.all()

define void @hardware_loop_probe(ptr addrspace(6) %0, i64 %1) #0 {
  %3 = getelementptr float, ptr addrspace(6) %0, i64 1024
  %4 = getelementptr float, ptr addrspace(6) %0, i64 2048
  br label %5

5:                                                ; preds = %aivscope.latch, %2
  %6 = phi i64 [ %27, %aivscope.latch ], [ 0, %2 ]
  %7 = icmp slt i64 %6, 1
  br i1 %7, label %8, label %28

8:                                                ; preds = %5
  %9 = call <256 x i1> @llvm.hivm.pset.b32(i32 0)
  br label %10

10:                                               ; preds = %13, %8
  %11 = phi i64 [ %25, %13 ], [ 0, %8 ]
  %12 = icmp slt i64 %11, %1
  br i1 %12, label %13, label %26

13:                                               ; preds = %10
  %14 = mul i64 %11, 64
  %15 = mul i64 %14, 4
  %16 = getelementptr i8, ptr addrspace(6) %0, i64 %15
  %17 = call <64 x float> @llvm.hivm.vldsx1.v64f32(ptr addrspace(6) %16, i32 0, i32 0, i32 0)
  %18 = mul i64 %14, 4
  %19 = getelementptr i8, ptr addrspace(6) %3, i64 %18
  %20 = call <64 x float> @llvm.hivm.vldsx1.v64f32(ptr addrspace(6) %19, i32 0, i32 0, i32 0)
  %21 = call <64 x float> @llvm.hivm.vadd.s.x.v64f32(<64 x float> %17, <64 x float> %20, <256 x i1> %9)
  %22 = call <64 x float> @llvm.hivm.vmul.s.x.v64f32(<64 x float> %21, <64 x float> %20, <256 x i1> %9)
  %23 = mul i64 %14, 4
  %24 = getelementptr i8, ptr addrspace(6) %4, i64 %23
  call void @llvm.hivm.vstsx1.v64f32(<64 x float> %22, ptr addrspace(6) %24, i32 0, i32 2, i32 0, <256 x i1> %9)
  call void @llvm.hivm.mem.bar.vv.all()
  %25 = add i64 %11, 1
  br label %10

26:                                               ; preds = %10
  br label %aivscope.latch

aivscope.latch:                                   ; preds = %26
  %27 = add i64 %6, 1
  br label %5, !llvm.loop !1

28:                                               ; preds = %5
  ret void
}

attributes #0 = { "target-cpu"="dav-c310-vec" "target-features"="+ATOMIC,+ArchV130,+AregRedefinable,+ArithmeticBf16,+AtomicForB8 ,+F8e4m3,+F8e5m2,+F8e8m0,+FFTSBlk,+Fp4e1m2x2,+Fp4e2m1x2,+LDExtRefine,+MOVX8,+SPR7bits,+SyncV,+dav-c310-vec" }

!llvm.module.flags = !{!0}
!hivm.annotations = !{}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = distinct !{!1, !2}
!2 = !{!"llvm.loop.aivector_scope"}
