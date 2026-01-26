#!/bin/bash

source .venv/bin/activate

export LIBGL_ALWAYS_SOFTWARE=1
export QT_X11_NO_MITSHM=1
export QT_QUICK_BACKEND=software
export QT_XCB_GL_INTEGRATION=none

# decord 패키지의 내장 libxcb가 시스템 라이브러리와 충돌하여 Segmentation fault를 유발합니다.
# 이를 해결하기 위해 내장 라이브러리의 이름을 변경하여 시스템 라이브러리를 사용하도록 합니다.
DECORD_LIBS=".venv/lib/python3.12/site-packages/decord.libs"

if [ -d "$DECORD_LIBS" ]; then
    # 1. 중복 백업(.bak.bak)된 파일이 있다면 .bak으로 복구합니다.
    for f in "$DECORD_LIBS"/libxcb-*.so.*.bak.bak; do
        [ -f "$f" ] && mv "$f" "${f%.bak}"
    done

    # 2. .bak 파일이 있는 경우, 해당 원본 위치에 올바른 시스템 라이브러리 링크를 생성합니다.
    for bak in "$DECORD_LIBS"/libxcb-*.so.*.bak; do
        if [ -f "$bak" ]; then
            target="${bak%.bak}"
            filename=$(basename "$target")
            
            # 파일명에 따라 매핑할 시스템 라이브러리 결정
            target_sys_lib=""
            if [[ "$filename" == libxcb-shape-* ]]; then target_sys_lib="libxcb-shape.so.0";
            elif [[ "$filename" == libxcb-shm-* ]]; then target_sys_lib="libxcb-shm.so.0";
            elif [[ "$filename" == libxcb-xfixes-* ]]; then target_sys_lib="libxcb-xfixes.so.0";
            elif [[ "$filename" == libxcb-* ]]; then target_sys_lib="libxcb.so.1"; fi

            if [ -n "$target_sys_lib" ]; then
                sys_lib_path=""
                for path in "/usr/lib/x86_64-linux-gnu/$target_sys_lib" "/lib/x86_64-linux-gnu/$target_sys_lib"; do
                    [ -e "$path" ] && sys_lib_path="$path" && break
                done
                [ -n "$sys_lib_path" ] && ln -sf "$sys_lib_path" "$target"
            fi
        fi
    done

    # 3. 백업되지 않은 원본 파일이 있다면 백업하고 링크를 생성합니다.
    for lib in "$DECORD_LIBS"/libxcb-*.so.*; do
        if [[ "$lib" != *.bak ]] && [ ! -L "$lib" ]; then
            filename=$(basename "$lib")
            
            target_sys_lib=""
            if [[ "$filename" == libxcb-shape-* ]]; then target_sys_lib="libxcb-shape.so.0";
            elif [[ "$filename" == libxcb-shm-* ]]; then target_sys_lib="libxcb-shm.so.0";
            elif [[ "$filename" == libxcb-xfixes-* ]]; then target_sys_lib="libxcb-xfixes.so.0";
            elif [[ "$filename" == libxcb-* ]]; then target_sys_lib="libxcb.so.1"; fi

            if [ -n "$target_sys_lib" ]; then
                sys_lib_path=""
                for path in "/usr/lib/x86_64-linux-gnu/$target_sys_lib" "/lib/x86_64-linux-gnu/$target_sys_lib"; do
                    [ -e "$path" ] && sys_lib_path="$path" && break
                done
                
                if [ -n "$sys_lib_path" ]; then
                    mv "$lib" "$lib.bak"
                    ln -sf "$sys_lib_path" "$lib"
                fi
            fi
        fi
    done
fi

python main.py
