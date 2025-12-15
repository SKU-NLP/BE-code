package com.app.NLP.global.template;

import com.app.NLP.global.exception.Error;
import com.app.NLP.global.exception.Success;
import lombok.AccessLevel;
import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.RequiredArgsConstructor;

@Getter
@RequiredArgsConstructor(access = AccessLevel.PRIVATE)
@AllArgsConstructor(access = AccessLevel.PRIVATE)
@NoArgsConstructor(force = true, access = AccessLevel.PRIVATE)
public class RspTemplate<T> {

    private final int statusCode;
    private final String message;
    private final T data;

    // ===== Success =====
    public static RspTemplate<Void> success(Success success) {
        return new RspTemplate<>(success.getHttpStatusCode(), success.getMessage(), null);
    }

    public static <T> RspTemplate<T> success(Success success, T data) {
        return new RspTemplate<>(success.getHttpStatusCode(), success.getMessage(), data);
    }

    // ===== Error =====
    public static RspTemplate<Void> error(Error error) {
        return new RspTemplate<>(error.getErrorCode(), error.getMessage(), null);
    }

    public static RspTemplate<Void> error(Error error, String message) {
        return new RspTemplate<>(error.getErrorCode(), message, null);
    }

    public static RspTemplate<Void> error(int customCode, String message) {
        return new RspTemplate<>(customCode, message, null);
    }
}
