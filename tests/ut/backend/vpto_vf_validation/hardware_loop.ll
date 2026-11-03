; ModuleID = 'ptoas.hivm.official.vector'
source_filename = "ptoas.hivm.official.vector"

declare <256 x i1> @llvm.hivm.pset.b32(i32)

declare { <64 x float>, ptr addrspace(6) } @llvm.hivm.vldsx1.post.v64f32(ptr addrspace(6), i32, i32, i32)

declare <64 x float> @llvm.hivm.vadd.s.x.v64f32(<64 x float>, <64 x float>, <256 x i1>)

declare <64 x float> @llvm.hivm.vmul.s.x.v64f32(<64 x float>, <64 x float>, <256 x i1>)

declare ptr addrspace(6) @llvm.hivm.vstsx1.post.v64f32(<64 x float>, ptr addrspace(6), i32, i32, i32, <256 x i1>)

define void @hardware_loop_probe(ptr addrspace(6) %0) #0 {
  %2 = getelementptr float, ptr addrspace(6) %0, i64 1024
  %3 = getelementptr float, ptr addrspace(6) %0, i64 2048
  br label %4

4:                                                ; preds = %aivscope.latch, %1
  %5 = phi i64 [ %30, %aivscope.latch ], [ 0, %1 ]
  %6 = icmp slt i64 %5, 1
  br i1 %6, label %7, label %31

7:                                                ; preds = %4
  %8 = getelementptr float, ptr addrspace(6) %0, i64 0
  %9 = getelementptr float, ptr addrspace(6) %2, i64 0
  %10 = getelementptr float, ptr addrspace(6) %3, i64 0
  %11 = call <256 x i1> @llvm.hivm.pset.b32(i32 0)
  br label %12

12:                                               ; preds = %18, %7
  %13 = phi i16 [ %28, %18 ], [ 0, %7 ]
  %14 = phi ptr addrspace(6) [ %21, %18 ], [ %8, %7 ]
  %15 = phi ptr addrspace(6) [ %24, %18 ], [ %9, %7 ]
  %16 = phi ptr addrspace(6) [ %27, %18 ], [ %10, %7 ]
  %17 = icmp slt i16 %13, 16
  br i1 %17, label %18, label %29

18:                                               ; preds = %12
  %19 = call { <64 x float>, ptr addrspace(6) } @llvm.hivm.vldsx1.post.v64f32(ptr addrspace(6) %14, i32 256, i32 0, i32 1)
  %20 = extractvalue { <64 x float>, ptr addrspace(6) } %19, 0
  %21 = extractvalue { <64 x float>, ptr addrspace(6) } %19, 1
  %22 = call { <64 x float>, ptr addrspace(6) } @llvm.hivm.vldsx1.post.v64f32(ptr addrspace(6) %15, i32 256, i32 0, i32 1)
  %23 = extractvalue { <64 x float>, ptr addrspace(6) } %22, 0
  %24 = extractvalue { <64 x float>, ptr addrspace(6) } %22, 1
  %25 = call <64 x float> @llvm.hivm.vadd.s.x.v64f32(<64 x float> %20, <64 x float> %23, <256 x i1> %11)
  %26 = call <64 x float> @llvm.hivm.vmul.s.x.v64f32(<64 x float> %25, <64 x float> %23, <256 x i1> %11)
  %27 = call ptr addrspace(6) @llvm.hivm.vstsx1.post.v64f32(<64 x float> %26, ptr addrspace(6) %16, i32 256, i32 2, i32 1, <256 x i1> %11)
  %28 = add i16 %13, 1
  br label %12

29:                                               ; preds = %12
  br label %aivscope.latch

aivscope.latch:                                   ; preds = %29
  %30 = add i64 %5, 1
  br label %4, !llvm.loop !1

31:                                               ; preds = %4
  ret void
}

attributes #0 = { "target-cpu"="dav-c310-vec" "target-features"="+ATOMIC,+ArchV130,+AregRedefinable,+ArithmeticBf16,+AtomicForB8 ,+F8e4m3,+F8e5m2,+F8e8m0,+FFTSBlk,+Fp4e1m2x2,+Fp4e2m1x2,+LDExtRefine,+MOVX8,+SPR7bits,+SyncV,+dav-c310-vec" }

!llvm.module.flags = !{!0}
!hivm.annotations = !{}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = distinct !{!1, !2}
!2 = !{!"llvm.loop.aivector_scope"}
